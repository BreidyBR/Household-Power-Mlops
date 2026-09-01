from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import hashlib
import json
from datetime import datetime, timezone


# Ruta raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directorio donde se almacenarán los datos crudos
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Archivo final esperado
RAW_FILE = RAW_DIR / "household_power_consumption.txt"

# Metadata
METADATA_FILE = RAW_DIR / "ingestion_metadata.json"

# Archivo ZIP temporal
ZIP_FILE = RAW_DIR / "household_power_consumption.zip"

# Fuente oficial del dataset en UCI
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)


def download_dataset():
    """Descarga el dataset desde UCI si todavía no existe."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_FILE.exists():
        print("El dataset raw ya existe. No se descargará nuevamente.")
        return

    print("Descargando dataset desde UCI...")
    urlretrieve(DATA_URL, ZIP_FILE)

    print("Descarga completada.")
    print(f"Archivo descargado: {ZIP_FILE}")


def extract_dataset():
    """Extrae el archivo original desde el ZIP descargado."""

    if RAW_FILE.exists():
        print("El archivo raw ya está disponible.")
        return

    if not ZIP_FILE.exists():
        raise FileNotFoundError(
            "No se encontró el archivo ZIP necesario para la extracción."
        )

    print("Extrayendo dataset...")

    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extract("household_power_consumption.txt", RAW_DIR)

    print("Extracción completada.")


def validate_raw_file():
    """Realiza validaciones básicas sobre el archivo raw."""

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            "La ingesta falló: el archivo raw no fue generado."
        )

    file_size = RAW_FILE.stat().st_size

    if file_size == 0:
        raise ValueError(
            "La ingesta falló: el archivo raw está vacío."
        )

    print("Validación básica del archivo completada.")
    print(f"Tamaño del archivo: {file_size / (1024 ** 2):.2f} MB")


def calculate_sha256(file_path):
    """Calcula el hash SHA-256 de un archivo."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()

def count_records(file_path):
    """Cuenta los registros de datos sin incluir la cabecera."""

    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        total_lines = sum(1 for _ in file)

    # Se resta 1 porque la primera línea corresponde a la cabecera
    return max(total_lines - 1, 0)


def create_metadata():
    """Genera metadata reproducible sobre el dataset ingerido."""

    file_size = RAW_FILE.stat().st_size
    file_hash = calculate_sha256(RAW_FILE)
    record_count = count_records(RAW_FILE)

    metadata = {
        "dataset": "Individual Household Electric Power Consumption",
        "source": DATA_URL,
        "ingestion_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file": RAW_FILE.name,
        "records": record_count,
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024 ** 2), 2),
        "sha256": file_hash,
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    print(f"Metadata guardada en: {METADATA_FILE}")
    print(f"Registros detectados: {record_count}")
    print(f"SHA-256: {file_hash}")


def main():
    """Punto de entrada del proceso de ingesta."""

    print("=== INICIO DEL PIPELINE DE INGESTA ===")

    download_dataset()
    extract_dataset()
    validate_raw_file()
    create_metadata()

    print(f"Dataset disponible en: {RAW_FILE}")
    print("=== INGESTA COMPLETADA ===")


if __name__ == "__main__":
    main()