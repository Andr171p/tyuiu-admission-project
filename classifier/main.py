from typing import Literal

from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from sklearn.pipeline import Pipeline
from fastapi import FastAPI, status
from pydantic import BaseModel, Field, ValidationError, field_validator

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"  # Пред-обученные модели

# Доступные направления подготовки
DIRECTIONS = [
    "01.03.02 Прикладная математика и информатика",
    "02.03.01 Математика и компьютерные науки",
    "05.03.01 Геология",
    "07.03.01 Архитектура",
    "07.03.03 Дизайн архитектурной среды",
    "08.03.01 Строительство",
    "08.05.00 Техника и технологии строительства",
    "09.03.00 Информатика и вычислительная техника",
    "09.03.02 Информационные системы и технологии",
    "12.03.01 Приборостроение",
    "12.03.04 Биотехнические системы и технологии",
    "13.03.01 Теплоэнергетика и теплотехника",
    "13.03.02 Электроэнергетика и электротехника",
    "15.03.01 Машиностроение",
    "15.03.04 Автоматизация технологических процессов и производств",
    "15.03.06 Мехатроника и робототехника",
    "18.03.00 Химические технологии",
    "18.03.01 Химическая технология",
    "20.03.01 Техносферная безопасность",
    "21.03.01 Нефтегазовое дело",
    "21.03.02 Землеустройство и кадастры",
    "21.05.00 Прикладная геология, горное дело, нефтегазовое дело и геодезия",
    "21.05.01 Прикладная геодезия",
    "21.05.02 Прикладная геология",
    "21.05.06 Нефтегазовые техника и технологии",
    "23.03.01 Технология транспортных процессов",
    "23.03.03 Эксплуатация транспортно-технологических машин и комплексов",
    "23.05.01 Наземные транспортно-технологические средства",
    "27.03.00 Управление в технических системах",
    "27.03.03 Системный анализ и управление",
    "27.03.04 Управление в технических системах",
    "42.03.01 Реклама и связи с общественностью",
    "43.03.00 Сервис и туризм",
]

encoder_gender = joblib.load(MODELS_DIR / "encoder_gender.joblib")
encoder_year = joblib.load(MODELS_DIR / "encoder_year.joblib")
standard_scaler = joblib.load(MODELS_DIR / "standard_scaler.joblib")
one_hot_encoder = joblib.load(MODELS_DIR / "one_hot_encoder.joblib")
classifier = joblib.load(MODELS_DIR / "classifier.joblib")

pipeline = Pipeline([
    ("transform", Pipeline([
        ("encode_gender", encoder_gender),
        ("encode_year", encoder_year),
        ("one_hot_encode", one_hot_encoder),
        ("scale", standard_scaler)
    ])),
    ("classify", classifier)
])


class ApplicantData(BaseModel):
    """Запрос для предсказания вероятности поступления абитуриента"""

    year: int = Field(ge=2019, le=2024)
    gender: Literal["male", "female"]
    gpa: float = Field(ge=3.0, le=5.0)
    points: int = Field(ge=0, le=310)
    direction: str

    @field_validator("direction")
    def validate_direction(cls, direction: str) -> str:
        if direction not in DIRECTIONS:
            raise ValueError(f"Unsupported direction `{direction}`!")
        return direction


def applicants_to_df(applicants: list[ApplicantData]) -> pd.DataFrame:
    return pd.DataFrame([applicant.model_dump() for applicant in applicants])


app = FastAPI(
    title="Сервис для предсказания вероятности поступления",
    description="""\
    ML модель бинарной классификации обученная на данных за 10+ лет
    """,
    version="0.1.0"
)


@app.get(
    path="/available-directions",
    status_code=status.HTTP_200_OK,
    response_model=list[str],
    summary="Доступные направления подготовки"
)
async def get_directions() -> list[str]:
    return DIRECTIONS


@app.post(
    path="/predict",
    status_code=status.HTTP_200_OK,
    response_model=list[float],
    summary="Получить вероятность поступления"
)
def predict(applicants: list[ApplicantData]) -> list[float]:
    applicants_df = pd.DataFrame([applicant.model_dump() for applicant in applicants])
    probas = pipeline.predict_proba(applicants_df)
    return [float(proba[-1]) for proba in probas]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
