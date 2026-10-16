"""Encrypted PDFs made by an independent tool (ADR-0001 Amendment 1).

The fixtures were made with qpdf 12.4.2 from a two-page PDF, with an empty user password
and copying and printing forbidden, as published reports often are:

    qpdf --encrypt --user-password= --owner-password=owner --bits=128 --use-aes=n \\
         --print=none --extract=n --allow-weak-crypto -- plain.pdf rc4-128.pdf
    ... --bits=128 --use-aes=y ...  -> aes-128.pdf
    ... --bits=256 ...              -> aes-256.pdf
    qpdf --encrypt --user-password=secret --owner-password=owner --bits=256 -- \\
         plain.pdf aes-256-password.pdf
"""

import importlib.util
import shutil
from pathlib import Path

import pypdf
import pytest
from pypdf.errors import DependencyError

from askpdf.chunking import ChunkParams
from askpdf.extract.base import ExtractionError
from askpdf.extract.pypdf_ext import PypdfExtractor
from askpdf.indexer import UNREADABLE_MTIME, index_folder
from askpdf.store import Store

FIXTURES = Path(__file__).parent / "fixtures" / "encrypted"
EMPTY_PASSWORD = ["rc4-128.pdf", "aes-128.pdf", "aes-256.pdf"]


@pytest.mark.parametrize("name", EMPTY_PASSWORD)
def test_empty_password_files_are_read(name: str) -> None:
    pages = PypdfExtractor().extract(FIXTURES / name)
    assert [p.text.strip() for p in pages] == [
        "Encrypted fixture page one.",
        "Second page of the fixture.",
    ]


def test_a_file_with_a_real_password_is_skipped_with_its_reason() -> None:
    with pytest.raises(ExtractionError, match="encrypted with a password") as info:
        PypdfExtractor().extract(FIXTURES / "aes-256-password.pdf")
    assert not info.value.retry  # nothing on this machine will change that


@pytest.mark.skipif(
    importlib.util.find_spec("cryptography") is not None,
    reason="pypdf prefers cryptography when installed (eval group); CI jobs do not have it",
)
def test_pycryptodome_is_the_backend_pypdf_uses() -> None:
    # pypdf only exposes this through a private module. Without the check, a job with
    # `cryptography` installed would pass the AES tests without touching pycryptodome.
    from pypdf._crypt_providers import crypt_provider

    assert crypt_provider[0] == "pycryptodome"


def test_index_reads_encrypted_files_and_reports_the_password_protected_one(
    tmp_path: Path,
) -> None:
    root = tmp_path / "folder"
    shutil.copytree(FIXTURES, root)
    with Store(tmp_path / "index.db") as store:
        report = index_folder(root, store, PypdfExtractor(), ChunkParams())
    assert sorted(report.added) == sorted(EMPTY_PASSWORD)
    assert report.failed == {"aes-256-password.pdf": "encrypted with a password"}


def test_without_a_crypto_backend_only_the_aes_file_fails_and_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "folder"
    root.mkdir()
    for name in ["rc4-128.pdf", "aes-256.pdf"]:
        shutil.copy(FIXTURES / name, root / name)
    real_decrypt = pypdf.PdfReader.decrypt

    def decrypt(self: pypdf.PdfReader, password: str | bytes) -> object:
        if "/AESV3" in str(self.trailer["/Encrypt"].get_object()):
            raise DependencyError("cryptography>=3.1 is required for AES algorithm")
        return real_decrypt(self, password)

    monkeypatch.setattr(pypdf.PdfReader, "decrypt", decrypt)
    with Store(tmp_path / "index.db") as store:
        report = index_folder(root, store, PypdfExtractor(), ChunkParams())
        row = store.files()["aes-256.pdf"]
        assert report.added == ["rc4-128.pdf"]
        assert report.failed == {
            "aes-256.pdf": "AES-encrypted; askpdf's crypto backend (pycryptodome) is not available"
        }
        assert row.mtime_ns == UNREADABLE_MTIME  # retried next run, not left as failed

        monkeypatch.setattr(pypdf.PdfReader, "decrypt", real_decrypt)
        again = index_folder(root, store, PypdfExtractor(), ChunkParams())
    assert again.added == ["aes-256.pdf"] and not again.failed


def test_running_out_of_memory_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise MemoryError

    monkeypatch.setattr(pypdf, "PdfReader", boom)
    with pytest.raises(ExtractionError, match="out of memory") as info:
        PypdfExtractor().extract(FIXTURES / "aes-128.pdf")
    assert info.value.retry
