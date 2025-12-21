import streamlit as st
from core.generator.prompt import system_prompt
from core.pipeline import gen_pipeline
import pandas as pd
from io import BytesIO

st.title("</> Title Generator")

with st.expander("📘 Инструкция по использованию"):
    st.markdown(
        """
        ### 📘 Инструкция по использованию
        
        - Загрузите файл с расширением xlsx
        - Файл должен содержать 3 поля: Page, Query, Volume:
            - Page - URL страницы
            - Query - запрос
            - Volume - частотность
            
        В настройках указаны режимы:
        - GPT-режим: отправляет фразы в GPT с промтом "как есть".
        - Алгоритмический режим: действует аналогично, но выполняется предобработка, которая позволяет
        автоматически определить основной запрос и другие части title, что может улучшить качество генерации.
        
        Выбирая данный режим нужно добавить в промпт следующий текст:
        
        ```
        Формат словаря:
            - head_phrase: главный запрос страницы, должен стоять в начале title
            - mandatory_phrases: фразы, которые ОБЯЗАТЕЛЬНО должны присутствовать (можно использовать синонимы)
            - optional_phrases: фразы, которые желательно включить, но не критично
            - synonyms: список синонимов для фраз, которые можно использовать для вариативности
        ```
        """
    )

with st.form("input_form"):
    strategy = st.selectbox("Выберите режим", ("GPT", "Algo"))
    model = st.selectbox("Укажите модель генерации ТЗ",
                         ("mistralai/devstral-2512:free", "openai/gpt-oss-20b:free", "xiaomi/mimo-v2-flash:free",
                          "tngtech/deepseek-r1t2-chimera:free"))
    prompt = st.text_area(label="Текст промта", value=system_prompt, height=700)
    input_file = st.file_uploader("Вставте данные в формате xlsx")
    submitted = st.form_submit_button("Сгенерировать тексты title")

if submitted:
    with st.spinner("Анализ выполняется, подождите..."):
        try:
            content = pd.read_excel(input_file)
            new_data = gen_pipeline(content, prompt, model, strategy)

            output = BytesIO()
            new_data.to_excel(output, index=False)
            st.download_button(
                label="Скачать результат генерации",
                data=output.getvalue(),
                file_name="title_content.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Произошла ошибка: {e}")
