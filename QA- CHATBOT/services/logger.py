# =========================
# IMPORTS
# =========================

import os
import json

from datetime import datetime

# =========================
# LOG DIRECTORY
# =========================

LOG_DIR = "logs"

LOG_FILE = os.path.join(

    LOG_DIR,

    "chatbot_logs.json"
)


# =========================
# CREATE LOG FOLDER
# =========================

os.makedirs(

    LOG_DIR,

    exist_ok=True
)


# =========================
# SAVE LOG ENTRY
# =========================

def save_log(log_entry):

    logs = []

    if os.path.exists(LOG_FILE):

        try:

            with open(

                LOG_FILE,

                "r",

                encoding="utf-8"
            ) as f:

                logs = json.load(f)

        except:

            logs = []

    logs.append(log_entry)

    with open(

        LOG_FILE,

        "w",

        encoding="utf-8"
    ) as f:

        json.dump(

            logs,

            f,

            indent=4,

            ensure_ascii=False
        )

        