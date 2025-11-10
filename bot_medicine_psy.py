# -*- coding: utf-8 -*-
"""
Телеграм-бот для расчёта стартовой дозы карбоната лития по мета-модели:
вход: пол, вес, рост, GFR (± целевой уровень)
выход: FFM, CL, стартовая суточная доза (элемент. Li и Li2CO3), деление на приёмы.

Команда:
/lithium sex=male weight=75 height=175 gfr=120 target=0.8 split=2 step=150
"""

import os
import math
from typing import Literal, Dict

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from pytz import timezone

Sex = Literal["male", "female"]

# -----------------------------
# Вспомогательные расчёты
# -----------------------------

def bmi(weight_kg: float, height_cm: float) -> float:
    """ИМТ = масса (кг) / рост^2 (м^2)."""
    h_m = height_cm / 100.0
    return weight_kg / (h_m ** 2)


def bayesian_dose_update(D_t: float, C_target: float, C_t: float,
                         omega: float = 0.30, S: float = 1.0, sigma: float = 0.10) -> float:
    """
    Упрощённое байесовское обновление дозы по TDM:
    K = (ω^2 * S^2) / (ω^2 * S^2 + σ^2)
    D_{t+1} = D_t * (C_target / C_t)^K
    - D_t: текущая суточная доза (в тех же единицах, что и вернёте)
    - C_target, C_t: концентрации в одних и тех же единицах (мЭкв/л или мг/л элементарного Li)
    - omega ~ межиндивид. вариабельность (CV≈30% ⇒ 0.30)
    - sigma ~ ошибка измерения (например, 0.1 мг/л)
    - S = 1 при пропорц. связи доза→концентрация
    """
    K = (omega**2 * S**2) / (omega**2 * S**2 + sigma**2)
    return D_t * (C_target / C_t) ** K


def ffm_janmahasatian(sex: Sex, weight_kg: float, height_cm: float) -> float:
    """
    Безжировая масса (FFM) по Janmahasatian 2005 (кг):
      муж.:  FFM = 9270*W / (6680 + 216*BMI)
      жен.:  FFM = 9270*W / (8780 + 244*BMI)
    """
    B = bmi(weight_kg, height_cm)
    if sex.lower() == "male":
        return 9270.0 * weight_kg / (6680.0 + 216.0 * B)
    else:
        return 9270.0 * weight_kg / (8780.0 + 244.0 * B)


def lithium_params_from_covariates(sex: Sex, weight_kg: float, height_cm: float, gfr_ml_min: float) -> Dict[str, float]:
    """
    Параметры лития (аппаратные) по мета-модели Lereclus et al. 2024:
      CL/F (л/ч) = 0.0734 + 0.117*(GFR/90) + 1.01*(FFM/50)
      V1/F=22.1 л, V2=3.35 л, Q=0.42 л/ч, Ka=0.62 ч^-1 (для справки)
    """
    ffm = ffm_janmahasatian(sex, weight_kg, height_cm)
    CL = 0.0734 + 0.117 * (gfr_ml_min / 90.0) + 1.01 * (ffm / 50.0)  # л/ч
    return dict(CL=CL, V1=22.1, V2=3.35, Q=0.42, Ka=0.62, FFM=ffm)


def lithium_mEq_to_mg_per_L(mEq_per_L: float) -> float:
    """мЭкв/л → мг/л (элементарный Li): 1 мЭкв/л = 6.94 мг/л."""
    return mEq_per_L * 6.94


def mg_Li_to_mg_Li2CO3(mg_elemental_li: float) -> float:
    """мг элементарного Li → мг Li2CO3. Массовая доля Li ≈ 18.8%."""
    return mg_elemental_li / 0.188


def lithium_starting_daily_dose(CL_L_per_h: float, C_target_mg_per_L: float) -> float:
    """
    Стартовая суточная доза по стац. правилу: Dose_day = 24 * CL * C_target.
    На выходе — мг/сут элементарного лития.
    """
    return 24.0 * CL_L_per_h * C_target_mg_per_L


def round_to_step(value_mg: float, step_mg: float) -> float:
    """Округление до ближайшего шага таблетки (например, 150 мг)."""
    if step_mg <= 0:
        return value_mg
    return round(value_mg / step_mg) * step_mg


# -----------------------------
# Парсер параметров из сообщения
# -----------------------------

def parse_kv_args(text: str) -> Dict[str, str]:
    """
    Разбирает строку вида: 'sex=male weight=75 height=175 gfr=120 target=0.8 split=2 step=150'
    Возвращает dict с ключами в нижнем регистре.
    """
    kv = {}
    for chunk in text.strip().split():
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            kv[k.strip().lower()] = v.strip()
    return kv


# -----------------------------
# Хэндлеры Telegram
# -----------------------------

HELP_TEXT = (
    "Пример команды:\n"
    "/lithium sex=male weight=75 height=175 gfr=120 target=0.8 split=2 step=150\n\n"
    "Пояснения:\n"
    "• sex: male|female\n"
    "• weight: масса, кг\n"
    "• height: рост, см\n"
    "• gfr: мл/мин\n"
    "• target: целевой литий в мЭкв/л (по умолчанию 0.8)\n"
    "• split: на сколько приёмов делить суточную дозу (по умолчанию 2)\n"
    "• step: шаг округления таблетки в мг Li2CO3 (по умолчанию 150)\n"
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я рассчитаю стартовую дозу карбоната лития по ковариатам.\n\n" + HELP_TEXT
    )


async def lithium_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        args_str = update.message.text.replace("/lithium", "", 1)
        kv = parse_kv_args(args_str)

        sex = kv.get("sex", "").lower()
        weight = float(kv.get("weight", "nan"))
        height = float(kv.get("height", "nan"))
        gfr = float(kv.get("gfr", "nan"))
        if sex not in ("male", "female") or math.isnan(weight) or math.isnan(height) or math.isnan(gfr):
            raise ValueError("Проверьте параметры: sex (male|female), weight, height, gfr.")

        target = float(kv.get("target", "0.8"))   # мЭкв/л
        split  = int(kv.get("split", "2"))
        step   = float(kv.get("step", "150"))

        # --- TDM (опционально) ---
        ct_meq = kv.get("ct_meq")                 # концентрация в мЭкв/л
        ct_mg  = kv.get("ct_mg")                  # концентрация в мг/л (элемент Li)
        omega  = float(kv.get("omega", "0.30"))
        sigma  = float(kv.get("sigma", "0.10"))
        S_par  = float(kv.get("s", "1.0"))        # ключи в kv — строчные!

        # --- Расчёты ---
        BMI = bmi(weight, height)
        params = lithium_params_from_covariates(sex, weight, height, gfr)
        ffm = params["FFM"]
        CL  = params["CL"]

        C_target_mg_L = lithium_mEq_to_mg_per_L(target)
        daily_li_mg   = lithium_starting_daily_dose(CL, C_target_mg_L)   # мг/сут элемент Li
        daily_li2co3  = mg_Li_to_mg_Li2CO3(daily_li_mg)                  # мг/сут Li2CO3

        per_intake_li2co3 = daily_li2co3 / max(split, 1)
        per_intake_li2co3_rounded = round_to_step(per_intake_li2co3, step)
        daily_li2co3_rounded = per_intake_li2co3_rounded * max(split, 1)

        # --- Байесовская коррекция по TDM (если задана) ---
        tdm_block = ""
        if ct_meq or ct_mg:
            if ct_meq is not None:
                C_t = lithium_mEq_to_mg_per_L(float(ct_meq))
                C_t_unit = f"{float(ct_meq):.2f} мЭкв/л"
            else:
                C_t = float(ct_mg)
                C_t_unit = f"{C_t:.2f} мг/л"

            daily_li_mg_new = bayesian_dose_update(
                D_t=daily_li_mg, C_target=C_target_mg_L, C_t=C_t,
                omega=omega, S=S_par, sigma=sigma
            )
            daily_li2co3_new = mg_Li_to_mg_Li2CO3(daily_li_mg_new)
            per_intake_new = daily_li2co3_new / max(split, 1)
            per_intake_new_rounded = round_to_step(per_intake_new, step)
            daily_new_rounded = per_intake_new_rounded * max(split, 1)

            K_val = (omega**2 * S_par**2) / (omega**2 * S_par**2 + sigma**2)
            tdm_block = (
                "\n🧭 Байесовская коррекция по TDM:\n"
                f"• Измеренная C_t: {C_t_unit}\n"
                f"• K = {K_val:.3f} (ω={omega:.2f}, σ={sigma:.2f}, S={S_par:.1f})\n"
                f"• Новая суточная доза (элемент Li): {daily_li_mg_new:.0f} мг/сут\n"
                f"• В Li₂CO₃: **{int(per_intake_new_rounded)} мг x {split} = {int(daily_new_rounded)} мг/сут**\n"
            )

        msg = (
            f"🧪 Расчёт стартовой дозы лития (мета-модель)\n"
            f"— Пол: {sex}\n"
            f"— Вес: {weight:.1f} кг, Рост: {height:.1f} см, GFR: {gfr:.0f} мл/мин\n"
            f"— BMI: {BMI:.1f} кг/м²; FFM (Janmahasatian): {ffm:.1f} кг\n"
            f"— CL (апп.): {CL:.3f} л/ч\n"
            f"— Цель: {target:.2f} мЭкв/л = {C_target_mg_L:.2f} мг/л (элемент Li)\n\n"
            f"Суточная доза (элемент Li): {daily_li_mг:.0f} мг/сут\n".replace("мг", "mg")  # только форматирование
            + (
                f"⇢ В Li₂CO₃: {daily_li2co3:.0f} мг/сут\n"
                f"⇢ Делим на {split} приёма: ~{per_intake_li2co3:.0f} мг/приём\n"
                f"⇢ С округлением {int(step)} мг: **{int(per_intake_li2co3_rounded)} мг x {split} = {int(daily_li2co3_rounded)} мг/сут**"
            )
            + tdm_block
        )

        await update.message.reply_text(msg, disable_web_page_preview=True)

    except Exception as e:
        await update.message.reply_text("⚠️ Ошибка: " + str(e))


# -----------------------------
# Запуск приложения
# -----------------------------

def main():
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("Переменная окружения TELEGRAM_TOKEN не задана.")

    app_tz = timezone("Europe/Amsterdam")  # pytz-tz, чтобы избежать ошибки APScheduler
    app = (
        Application.builder()
        .token(token)
        .timezone(app_tz)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("lithium", lithium_cmd))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
