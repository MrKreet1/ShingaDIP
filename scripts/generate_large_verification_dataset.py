from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "verification_data" / "large_check"
OUTPUT_FILE = OUTPUT_DIR / "operations_large_5000.csv"


COUNTERPARTIES = [
    "TOO Delta Supplies",
    "TOO North Office",
    "TOO Metro Parts",
    "TOO City Trade",
    "IP Audit Expert",
    "TOO Vision Market",
    "TOO Logistic Group",
    "TOO Prime Service",
]

DOCUMENT_TYPES = [
    "Счет-фактура",
    "Акт",
]

DESCRIPTIONS = [
    "Закупка канцелярских товаров",
    "Услуги сопровождения офиса",
    "Поставка расходных материалов",
    "Техническое обслуживание оборудования",
    "Маркетинговые услуги",
    "Транспортные услуги",
]

EMPLOYEES = [
    "Айгерим С.",
    "Нурлан Т.",
    "Марина Ж.",
    "Арман О.",
    "Дамир К.",
]

ACCOUNTS = ["3310", "7210", "2930"]


def build_row(index: int) -> list[str]:
    base_date = date(2026, 1, 1) + timedelta(days=index % 180)
    counterparty = COUNTERPARTIES[index % len(COUNTERPARTIES)]
    document_type = DOCUMENT_TYPES[index % len(DOCUMENT_TYPES)]
    description = DESCRIPTIONS[index % len(DESCRIPTIONS)]
    employee = EMPLOYEES[index % len(EMPLOYEES)]
    account = ACCOUNTS[index % len(ACCOUNTS)]

    amount = 45000 + (index % 37) * 7250
    vat = round(amount * 0.12, 2)
    document_number = f"LG-{index + 1000:05d}"

    if index % 125 == 0:
        amount = 2750000
        vat = 0
        description = "Срочная корректировка вручную"
        counterparty = "TOO Unknown Services"
        account = "7210"
        document_type = "Прочее"

    if index % 111 == 0:
        description = ""

    if index % 173 == 0 and index > 0:
        document_number = f"LG-{index + 999:05d}"

    if index % 289 == 0:
        counterparty = f"IP Rare Vendor {index // 289 + 1}"

    return [
        base_date.isoformat(),
        document_number,
        document_type,
        counterparty,
        str(amount),
        "KZT",
        str(vat),
        account,
        description,
        employee,
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    headers = [
        "Дата операции",
        "Номер документа",
        "Тип документа",
        "Контрагент",
        "Сумма",
        "Валюта",
        "НДС",
        "Счет учета",
        "Описание операции",
        "Ответственный сотрудник",
    ]

    with OUTPUT_FILE.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for index in range(5000):
            writer.writerow(build_row(index))

    print(f"Created {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
