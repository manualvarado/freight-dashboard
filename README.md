# 🚛 Freight Operations Dashboard

A comprehensive analytics dashboard for freight companies to track operational performance, carrier efficiency, and business metrics.

## 📊 Features

- **11 Interactive Visualizations**: From billing analysis to driver performance
- **Written Analysis**: Business insights and recommendations for each chart
- **CSV Upload**: Easy data import from your TMS system
- **Filtering**: Filter by dispatcher, carrier, trailer type
- **Export Options**: Download reports and analysis
- **Real-time Analytics**: Instant insights from your data

## 🚀 Quick Start

### Local Development
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/freight-dashboard.git
cd freight-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run main.py
```

### Deployment Options

#### 1. Streamlit Cloud (Recommended)
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect your repository
4. Deploy automatically

#### 2. Heroku
```bash
# Install Heroku CLI
heroku create your-freight-dashboard
git push heroku main
heroku open
```

## 📋 Expected CSV Format

Your CSV should include these columns:
- `LOAD ID`: Unique load identifier
- `DISPATCH NAME`: Name of the dispatcher
- `BROKER RATE (FC) [$`: Broker rate amount
- `DRIVER RATE [$]`: Driver pay amount
- `DRIVER NAME`: Name of the driver
- `TRAILER TYPE`: Type of trailer used
- `LOAD STATUS`: Status of the load (Booked, Delivered, etc.)
- `LOAD'S CARRIER COMPANY`: Carrier company name
- `FULL MILES TOTAL`: Total miles for the load
- `BD MARGIN`: BD margin field
- `PICK UP DATE`: Pickup date for time series analysis

## 📈 Dashboard Sections

### 1. Key Performance Indicators
- Total Valid Loads
- Total Billing
- Total Driver Pay
- B-Rate %

### 2. Enhanced Analytics
- **Carrier Performance**: Billing and efficiency analysis
- **Dispatcher Analysis**: Performance matrix and load distribution
- **Driver Insights**: Earnings analysis and heatmap patterns
- **Operational Metrics**: Trailer type profitability and weekly trends
- **Margin Analysis**: Distribution and time series analysis

### 3. Data Tables
- Dispatcher billing summary
- Carrier miles summary

### 4. Export Options
- Summary reports
- Dispatcher analysis
- Driver analysis

## 🔧 Configuration

### Environment Variables
```bash
# For production deployments
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Customization
- Modify `enhanced_visualizer.py` for new chart types
- Update `generate_chart_analysis()` for custom insights
- Add new filters in `main.py`

## 📊 Business Intelligence

The dashboard provides actionable insights:
- **Carrier Efficiency**: Revenue per mile analysis
- **Performance Gaps**: Dispatcher and driver comparisons
- **Operational Trends**: Weekly and seasonal patterns
- **Margin Optimization**: Pricing strategy recommendations
- **Capacity Planning**: Resource allocation insights

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues or questions:
- Create an issue on GitHub
- Contact: your-email@domain.com

---

**Built with ❤️ for the freight industry** 