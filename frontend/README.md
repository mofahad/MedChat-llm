# MedChat React client

The root README documents the verified Python backend and the simpler single-server source explorer.

For this optional React client, start the backend with `python app.py` from the repository root, then run `npm ci` and `npm start` in this folder. The API defaults to `http://127.0.0.1:5005`; set `REACT_APP_API_URL` to use another local backend.

Use `npm test -- --watchAll=false --runInBand` for behavior checks and `npm run build` for a production bundle. The interface returns project-note source excerpts, not generated medical advice.
