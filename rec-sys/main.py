from typing import Literal

import json
import logging
from pathlib import Path

import pandas as pd
import uvicorn
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, Query, status, HTTPException
from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, field_validator

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

STUDENTS_CSV = DATA_DIR / "Студенты_2014_2024.csv"
DIRECTIONS_JSON = DATA_DIR / "Направления_подготовки.json"
PASSING_SCORES_CSV = DATA_DIR / "Проходные_баллы_2014_2024.csv"
DIRECTIONS_MAP_JSON = DATA_DIR / "Маппинг_направлений_подготовки.json"

# Данные по студентам с 2014 по 2024 год
students_df = pd.read_csv(STUDENTS_CSV)

# Проходные баллы с 2014 по 2024 год
passing_scores_df = pd.read_csv(PASSING_SCORES_CSV)

# Справочник с информацией о направлениях подготовки
directions_data = json.loads(Path(DIRECTIONS_JSON).read_text(encoding="utf-8"))

# Маппинг наименований направлений подготовки
directions_map = json.loads(Path(DIRECTIONS_MAP_JSON).read_text(encoding="utf-8"))

RUSSIAN_TO_ENGLISH_SUBJECTS = {
    "Русский язык": "russian",
    "Математика": "math",
    "Обществознание": "social_science",
    "Физика": "physics",
    "Химия": "chemistry",
    "История": "history",
    "Информатика": "informatics",
}

# Колонки из CSV-файла, используемые для расчёта рекомендаций
FEATURES = [
    "Пол",
    "Ср. балл док-та об образовании",
    "Сумма баллов",
    "Обществознание",
    "Математика",
    "Информатика",
    "Русский язык",
    "Физика",
    "Химия",
    "История",
]


class Exam(BaseModel):
    """Экзамен ЕГЭ"""

    subject_name: Literal[
        "Русский язык",
        "Обществознание",
        "Математика",
        "Физика",
        "Химия",
        "История",
        "Информатика"
    ]
    points: int = Field(ge=0, le=100)

    @property
    def subject_code(self) -> str:
        return RUSSIAN_TO_ENGLISH_SUBJECTS[self.subject_name]


class ApplicantData(BaseModel):
    """Данные абитуриента для расчёта персональных рекомендаций"""

    gender: Literal["male", "female"]
    gpa: float = Field(ge=3.0, le=5.0)
    total_score: NonNegativeInt = Field(le=310)
    exams: list[Exam]

    @property
    def gender_code(self) -> Literal[0, 1]:
        return 1 if self.gender == "male" else 0

    @property
    def exam_codes_to_points(self) -> dict[str, int]:
        return {exam.subject_code: exam.points for exam in self.exams}

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "Пол": self.gender_code,
            "Ср. балл док-та об образовании": self.gpa,
            "Сумма баллов": self.total_score,
            "Обществознание": self.exam_codes_to_points.get("social_science", 0),
            "Математика": self.exam_codes_to_points.get("math", 0),
            "Информатика": self.exam_codes_to_points.get("informatics", 0),
            "Русский язык": self.exam_codes_to_points.get("russian", 0),
            "Физика": self.exam_codes_to_points.get("physics", 0),
            "Химия": self.exam_codes_to_points.get("chemistry", 0),
            "История": self.exam_codes_to_points.get("history", 0)
        }])


class EntranceExam(BaseModel):
    """Вступительный экзамен"""

    priority: PositiveInt = Field(ge=1, le=5)
    name: str
    min_score: NonNegativeInt


class PassingScore(BaseModel):
    """Проходной балл"""

    year: PositiveInt
    score: NonNegativeInt


class Direction(BaseModel):
    """Направление подготовки"""

    id: int
    title: str
    education_form: Literal[
        "ОФО",
        "ЗФО",
        "О-ЗФО",
        "ОФО; ЗФО"
    ]
    description: str
    entrance_exams: list[EntranceExam]
    passing_scores: list[PassingScore]

    @field_validator("title")
    def validate_title(cls, title: str) -> str:
        return directions_map[title]


app = FastAPI(
    title="Рекомендательная система направлений подготовки",
    description="""\
    Рекомендует наиболее подходящие направления подготовки основываясь на данных
    студентов с 2014 по 2024 год
    """,
    version="0.1.0",
)


@app.post(
    path="/recommendations",
    status_code=status.HTTP_200_OK,
    summary="Получение персональных рекомендаций"
)
def get_recommendations(
        applicant: ApplicantData,
        top_n: PositiveInt = Query(..., le=52, description="Количество рекомендаций")
) -> list[dict[str, int | str]]:
    applicant_df = applicant.to_df()
    combined_df = pd.concat([students_df[FEATURES], applicant_df], ignore_index=True)
    applicant_index = len(students_df)
    similarity_matrix = cosine_similarity(combined_df)
    similar_indices = similarity_matrix[applicant_index].argsort()[::-1][1:]
    similar_students = students_df.iloc[similar_indices][
        ["Направление подготовки", "ID"]
    ]
    unique_directions = similar_students.drop_duplicates(subset="ID")
    return [
        {
            "direction_id": direction["ID"],
            "direction_title": direction["Направление подготовки"]
        }
        for _, direction in unique_directions.head(top_n).iterrows()
    ]


@app.get(
    path="/directions/{direction_id}",
    status_code=status.HTTP_200_OK,
    response_model=Direction,
    summary="Получение направления подготовки"
)
def get_direction(direction_id: int) -> Direction:
    direction_data = next(
        (
            direction_data for direction_data in directions_data
            if direction_data["id"] == direction_id), None
    )
    if direction_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Direction not found"
        )
    entrance_exams = [
        EntranceExam(
            priority=exam["Приоритет"],
            name=exam["Наименование"],
            min_score=exam["Минимальный балл"],
        )
        for exam in direction_data["Вступительные испытания"]
    ]
    direction_passing_scores_df = passing_scores_df[passing_scores_df["ID"] == direction_id]
    passing_scores = [
        PassingScore(year=row["Год"], score=row["Сумма баллов"])
        for _, row in direction_passing_scores_df.iterrows()
    ]
    return Direction(
        id=direction_data["id"],
        education_form=direction_data["Форма обучения"],
        title=direction_data["Направление подготовки"],
        description=direction_data["Описание"],
        entrance_exams=entrance_exams,
        passing_scores=passing_scores,
    )


@app.get(
    path="/directions",
    status_code=status.HTTP_200_OK,
    response_model=Direction,
    summary="Получение направления по его наименованию"
)
def get_direction_by_title(
        title: str = Query(
            ...,
            description="Наименование",
            examples=["09.03.00 Информатика и вычислительная техника"]
        )
) -> Direction:
    direction_data = next(
        (
            direction_data
            for direction_data in directions_data
            if direction_data["Направление подготовки"] == title
        ),
        None,
    )
    if direction_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Direction with title `{title}` not found"
        )
    entrance_exams = [
        EntranceExam(
            priority=exam["Приоритет"],
            name=exam["Наименование"],
            min_score=exam["Минимальный балл"],
        )
        for exam in direction_data["Вступительные испытания"]
    ]
    direction_passing_scores_df = passing_scores_df[
        passing_scores_df["ID"] == direction_data["id"]
    ]
    passing_scores = [
        PassingScore(year=row["Год"], score=row["Сумма баллов"])
        for _, row in direction_passing_scores_df.iterrows()
    ]
    return Direction(
        id=direction_data["id"],
        education_form=direction_data["Форма обучения"],
        title=direction_data["Направление подготовки"],
        description=direction_data["Описание"],
        entrance_exams=entrance_exams,
        passing_scores=passing_scores,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app=app, host="0.0.0.0", port=8000)
