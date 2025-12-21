from core.generator.algo import build_roles
from core.generator.llm import generate_title


def gen_pipeline(data, prompt, model, strategy="GPT"):
    df = data.copy()
    unique_pages = df["Page"].unique()

    for p in unique_pages:
        queries = df[df["Page"] == p].iloc[:, 1:3]
        if strategy == "Algo":
            content = build_roles(queries)
        else:
            content = queries.copy()
        t = generate_title(content, prompt, model, strategy)
        df.loc[df["Page"] == p, "New_title"] = t

    return df
