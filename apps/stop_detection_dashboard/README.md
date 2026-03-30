# Stop Detection Dashboard (Flask + JS)

This dashboard is a conventional Python web app:

```bash
cd apps/stop_detection_dashboard
python app.py
```

Then open:

- http://127.0.0.1:5000

## Structure

- `app.py`: Flask backend and API endpoints
- `templates/index.html`: page shell
- `static/app.js`: primary UI logic and animation
- `static/styles.css`: styling

## Notes

- Uses NOMAD stop-detection algorithms on `nomad/data/gc_sample.csv`
- Frontend animation and rendering are JavaScript-driven (Leaflet + Plotly)
- If optional NOMAD dependencies are missing, the API returns a clear error
