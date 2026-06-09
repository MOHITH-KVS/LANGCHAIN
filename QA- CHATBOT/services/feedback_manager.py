# =========================
# IMPORTS
# =========================

import os
import json

from datetime import datetime


# =========================
# FEEDBACK FILE
# =========================

FEEDBACK_DIR = "feedback"

FEEDBACK_FILE = os.path.join(

    FEEDBACK_DIR,

    "feedback_logs.json"
)


# =========================
# CREATE DIRECTORY
# =========================

os.makedirs(

    FEEDBACK_DIR,

    exist_ok=True
)


# =========================
# SAVE FEEDBACK
# =========================

def save_feedback(

    question,

    feedback
):

    feedback_entry = {

        "timestamp": str(datetime.now()),

        "question": question,

        "feedback": feedback
    }

    feedback_data = []

    if os.path.exists(FEEDBACK_FILE):

        try:

            with open(

                FEEDBACK_FILE,

                "r",

                encoding="utf-8"
            ) as f:

                feedback_data = json.load(f)

        except:

            feedback_data = []

    feedback_data.append(

        feedback_entry
    )

    with open(

        FEEDBACK_FILE,

        "w",

        encoding="utf-8"
    ) as f:

        json.dump(

            feedback_data,

            f,

            indent=4,

            ensure_ascii=False
        )