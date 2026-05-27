========================================================================
ELEVATOR TRAFFIC FORECASTING HYBRID PIPELINE DASHBOARD
========================================================================

This interactive application simulates the full data science framework 
used to forecast vertical transit patterns in ultra-large skyscrapers. 
It covers SAITS Imputation, ARIMA linear profiling, and BiLSTMA+Attention 
residual feature mapping.

------------------------------------------------------------------------
PREREQUISITES
------------------------------------------------------------------------
Ensure you have Python 3.9 or higher installed on your machine.

------------------------------------------------------------------------
INSTALLATION & CONFIGURATION
------------------------------------------------------------------------
1. Move the 'app.py' and 'requirements.txt' files into a single directory
   on your system.

2. Open your terminal or command prompt and navigate to that directory:
   cd /path/to/your/directory

3. (Optional but recommended) Create and activate a clean virtual environment:
   python -m venv venv
   source venv/bin/activate       # On macOS/Linux
   .\venv\Scripts\activate        # On Windows

4. Install all the required analytical and charting dependencies:
   pip install -r requirements.txt

------------------------------------------------------------------------
RUNNING THE APPLICATION
------------------------------------------------------------------------
1. Execute the main application entrypoint via Streamlit:
   streamlit run app.py

2. The application will compile the assets locally. A browser window will
   automatically pop open at:
   http://localhost:8501

3. Use the dynamic sidebar panel to step through each isolated phase 
   of the hybrid architecture.
=======================================================================