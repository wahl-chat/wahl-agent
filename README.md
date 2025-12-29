# wahl-agent

This conversation agent introduces an interventionist style of conversation that promotes active thinking and processing for political education. Currently it is focused around currently elected German political parties in the parlament.

## Setup
To run this project locally follow these instructions:

0. Have Python's [poetry](https://python-poetry.org/docs/#installation) installed

1. Install dependencies:
```bash
poetry install
```

2. Set up your OpenAI API key as an environment variable and the path to the Firestore credential's JSON.
Create a `.env` file in the project root (follow .env.example):

## Running the Application

To run the application locally using debug flag (hot reloading) execute the following command:
```bash
poetry run flask --app src/controller.py run --debug
```

## License
This project is **source-available** under the **PolyForm Noncommercial 1.0.0** license.
- Free for **non-commercial** use (see LICENSE for permitted purposes)
- Share the license text and any `Required Notice:` lines when distributing
- Please contact us at info@wahl.chat to:

  - Inform us about your use case
  - Get access to assets required for a reference to wahl.chat on your project page
- Do not use the wahl.chat name or logo in your project without our permission
