# Optional hosting, without an API key

No hosted demo was created or verified in this repair. The localhost instructions in the main README are the tested path. The following Docker recipe is prepared for a future manual deployment; Docker and hosted execution have not been tested here.

1. Sign in to your own Hugging Face account in the browser and choose **New Space**.
2. Choose **Docker**, **Blank**, and the free **CPU Basic** hardware option if offered. Do not select a paid upgrade. Choose visibility deliberately; a public Space is visible to everyone.
3. In the Space's file editor, use the contents of `docs/SPACE_README.md` as its root `README.md`.
4. Upload `Dockerfile`, `requirements.txt`, `requirements-hosting.txt`, `app.py`, `data/demo_notes.json`, and `demo/index.html`, preserving the folders.
5. Wait for the build, then test the Model example and an unmatched query in the App tab. No API token or secret should be added.
6. Add the real Space URL to the GitHub README only after both checks work. Do not substitute a guessed URL.

The Dockerfile serves the project-note demo with Gunicorn. It does not deploy the archived BioMistral notebook or reproduce its research results.

Sources: [Hugging Face Docker Spaces](https://huggingface.co/docs/hub/en/spaces-sdks-docker), [Flask Gunicorn application factories](https://flask.palletsprojects.com/en/stable/deploying/gunicorn/).
