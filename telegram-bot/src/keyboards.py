from aiogram.types import ReplyKeyboardMarkup
from aiogram.utils.keyboard import ReplyKeyboardBuilder


def get_role_choice_kb() -> ReplyKeyboardMarkup:
    """Клавиатура для выбора роли пользователя"""

    builder = ReplyKeyboardBuilder()
    builder.button(text="Школьник")
    builder.button(text="Абитуриент")
    builder.button(text="Родитель")
    builder.adjust(1)
    return builder.as_markup(one_time_keyboard=True, resize_keyboard=True)
