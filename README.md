# CS163 – Dash App Template

A minimal [Plotly Dash](https://dash.plotly.com/) project template to get you started quickly.

## Project Structure

```
CS163/
├── app.py              # Main Dash application
├── requirements.txt    # Python dependencies
├── assets/
│   └── style.css       # Global stylesheet (auto-loaded by Dash)
└── README.md
```

## Setup

1. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:

   ```bash
   python app.py
   ```

4. Open your browser and navigate to <http://127.0.0.1:8050/>.

## Customization

- Edit `app.py` to change the layout, data, and callbacks.
- Add or modify CSS rules in `assets/style.css` to style your app.
- Place additional static files (images, fonts, JS) inside the `assets/` folder.