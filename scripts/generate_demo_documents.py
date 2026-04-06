from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "sample_data" / "documents"


DOCUMENTS = [
    {
        "file_name": "INV-1001.pdf",
        "lines": [
            "Invoice",
            "Document Number: INV-1001",
            "Document Date: 2026-03-01",
            "Seller: TOO Altyn Trade",
            "Amount: 125000 KZT",
            "Description: Office supplies purchase",
        ],
    },
    {
        "file_name": "INV-1002.pdf",
        "lines": [
            "Invoice",
            "Document Number: INV-1002",
            "Document Date: 2026-03-02",
            "Seller: TOO Altyn Trade",
            "Amount: 121500 KZT",
            "Description: Paper purchase",
        ],
    },
    {
        "file_name": "ACT-2001.pdf",
        "lines": [
            "Service Act",
            "Document Number: ACT-2001",
            "Document Date: 2026-03-04",
            "Seller: IP Sapa Service",
            "Amount: 98000 KZT",
            "Description: Office support services",
        ],
    },
    {
        "file_name": "INV-1004.pdf",
        "lines": [
            "Invoice",
            "Document Number: INV-1004",
            "Document Date: 2026-03-04",
            "Seller: TOO Vector Parts",
            "Amount: 470000 KZT",
            "Description: Printer spare parts",
        ],
    },
    {
        "file_name": "INV-1007.pdf",
        "lines": [
            "Invoice",
            "Document Number: INV-1007",
            "Document Date: 2026-03-08",
            "Seller: TOO Office Goods",
            "Amount: 79000 KZT",
            "Description: Office accessories",
        ],
    },
    {
        "file_name": "INV-1010.pdf",
        "lines": [
            "Invoice",
            "Document Number: INV-1010",
            "Document Date: 2026-03-10",
            "Seller: TOO New Counterparty",
            "Amount: 1500000 KZT",
            "Description: Consulting services",
        ],
    },
]


def escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(lines: list[str]) -> bytes:
    content_lines = ["BT", "/F1 12 Tf", "50 770 Td", "16 TL"]
    for index, line in enumerate(lines):
        prefix = "" if index == 0 else "T* "
        content_lines.append(f"{prefix}({escape_pdf_text(line)}) Tj")
    content_lines.append("ET")
    content_stream = "\n".join(content_lines).encode("latin-1")

    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
        f"5 0 obj << /Length {len(content_stream)} >> stream\n".encode("latin-1")
        + content_stream
        + b"\nendstream endobj",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)
        pdf.extend(b"\n")

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))

    trailer = (
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n"
    )
    pdf.extend(trailer.encode("latin-1"))
    return bytes(pdf)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for document in DOCUMENTS:
        pdf_bytes = build_pdf(document["lines"])
        output_path = OUTPUT_DIR / document["file_name"]
        output_path.write_bytes(pdf_bytes)
        print(f"Created {output_path}")


if __name__ == "__main__":
    main()

