import aiohttp
from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message, ReplyKeyboardRemove
from aiogram.utils.chat_action import ChatActionSender

from .db import create_user, get_user
from .keyboards import get_role_choice_kb
from .settings import settings

router = Router(name=__name__)


class UserForm(StatesGroup):
    """Форма для заполнения информации о пользователе"""

    in_role_choice = State()
    in_purpose_typing = State()


@router.message(CommandStart())
async def cb_start(message: Message, state: FSMContext) -> None:
    """Обработка команды `/start`"""

    user = await get_user(message.from_user.id)
    if user is None:
        await message.answer(
            text=(
                "<b>Добро пожаловать!</b>\n\n"
                "Для максимально качественных ответов, мне нужно"
                "немного о вас узнать.\n"
                "Это займёт не более <u>1 минуты</u>."
            ), reply_markup=get_role_choice_kb()
        )
        await state.set_state(UserForm.in_role_choice)
        return
    await message.answer(text="Давайте продолжим общение ...")
    await state.update_data(role=user.role, purpose=user.purpose)


@router.message(UserForm.in_role_choice, F.text)
async def process_role_choice(message: Message, state: FSMContext) -> None:
    """Обработка выбора роли пользователя"""

    await state.update_data(role=message.text)
    await message.reply(
        text=(
            "Отлично, теперь укажите свою цель.\n\n"
            "<i>Например:</i>\n"
            " - Поступить на бюджет\n"
            " - Найти подходящее направление"
            " - Подготовится к ЕГЭ\n\n"
            "<i>Напишите сообщением:</i>"
        )
    )
    await state.set_state(UserForm.in_purpose_typing)


@router.message(UserForm.in_purpose_typing, F.text)
async def process_purpose(message: Message, state: FSMContext) -> None:
    """Обработка введённой цели пользователя"""

    data = await state.get_data()
    await create_user(
        user_id=message.from_user.id,
        username=message.from_user.username,
        role=data["role"],
        purpose=message.text.strip()
    )
    await message.answer(
        text="Спасибо за уделённое время! Теперь вы можете задать интересующие вас вопросы",
        reply_markup=ReplyKeyboardRemove(),
    )
    await state.clear()
    await state.update_data(role=data["role"], purpose=message.text.strip())


@router.message(F.text)
async def handle_user_message(message: Message, state: FSMContext) -> None:
    """Обработка сообщения пользователя"""

    data = await state.get_data()
    role, purpose = data["role"], data["purpose"]
    timeout = aiohttp.ClientTimeout(total=120)
    payload = {
        "user_id": f"{message.from_user.id}",
        "role": role,
        "purpose": purpose,
        "text": message.text
    }
    headers = {"Content-Type": "application/json"}
    async with ChatActionSender.typing(chat_id=message.chat.id, bot=message.bot):
        async with aiohttp.ClientSession(
                base_url=settings.agent_base_url, timeout=timeout
        ) as session, session.post(
            url="/agent", json=payload, headers=headers
        ) as response:
            response.raise_for_status()
            data = await response.json()
        await message.reply(text=data["text"])
