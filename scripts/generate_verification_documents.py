from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "verification_data" / "document_check" / "documents"


PDF_DOCUMENTS = [
    {
        "file_name": "INV-3001.pdf",
        "lines": [
            "Invoice",
            "Document Number: INV-3001",
            "Document Date: 2026-04-01",
            "Seller: TOO Delta Supplies",
            "Amount: 158000 KZT",
            "Description: Office furniture supply",
        ],
    },
    {
        "file_name": "ACT-3002.pdf",
        "lines": [
            "Service Act",
            "Document Number: ACT-3002",
            "Document Date: 2026-04-03",
            "Seller: IP Audit Expert",
            "Amount: 92000 KZT",
            "Description: Internal audit services",
        ],
    },
    {
        "file_name": "INV-3003.pdf",
        "lines": [
            "Invoice",
            "Document Number: INV-3003",
            "Document Date: 2026-04-03",
            "Seller: TOO Media Group",
            "Amount: 300000 KZT",
            "Description: Marketing campaign",
        ],
    },
]


IMAGE_DOCUMENT = {
    "file_name": "IMG-3004.png",
    "lines": [
        "Invoice",
        "Document Number: IMG-3004",
        "Document Date: 2026-04-04",
        "Seller: TOO Vision Market",
        "Amount: 76000 KZT",
        "Description: Office consumables",
    ],
}


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


def build_image_document(file_name: str, lines: list[str]) -> None:
    image = Image.new("RGB", (1400, 900), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    y = 90
    for line in lines:
        draw.text((80, y), line, fill="black", font=font)
        y += 85

    image.save(OUTPUT_DIR / file_name)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for document in PDF_DOCUMENTS:
        output_path = OUTPUT_DIR / document["file_name"]
        output_path.write_bytes(build_pdf(document["lines"]))
        print(f"Created {output_path}")

    build_image_document(IMAGE_DOCUMENT["file_name"], IMAGE_DOCUMENT["lines"])
    print(f"Created {OUTPUT_DIR / IMAGE_DOCUMENT['file_name']}")


if __name__ == "__main__":
    main()
