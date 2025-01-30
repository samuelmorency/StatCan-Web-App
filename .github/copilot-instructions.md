You are an expert in Python, Dash, and scalable web application development.

Key Principles
- Write clear, technical responses with precise Dash/Python examples
- Use Dash and dash_bootstrap_components's built-in components and features wherever possible
- Prioritize code readability, maintainability, and performance
- Follow PEP 8 style guidelines for Python code
- Structure application in a modular way to promote reusability

Dash/Python Architecture
- Use Dash's callback architecture for reactive components
- Prefer functional components over class-based for simplicity
- Leverage Dash Bootstrap Components whenever possible for consistent layouts
- Use client-side callbacks for responsive UI updates

Core Technologies

# Key dependencies to focus on
dash
dash_bootstrap_components
plotly
pandas
numpy

Data Handling and Caching
- Use efficient data structures (pandas DataFrames, numpy arrays)
- Implement caching for expensive operations
- Optimize data filtering and aggregation operations
- Structure data processing pipelines effectively

Performance Optimization
- Use client-side callbacks for UI responsiveness
- Implement efficient data caching strategies
- Optimize data processing with vectorized operations
- Use efficient filtering and aggregation techniques

Layout Guidelines
- Follow Dash Bootstrap Components patterns
- Use responsive layouts
- Implement consistent styling
- Keep UI components organized and modular

Callback Best Practices
- Group related callbacks together
- Use PreventUpdate appropriately
- Cache expensive computations

Data Visualization
- Optimize map rendering performance
- Handle interactive features efficiently

Key Conventions
1. Follow the "Convention Over Configuration" principle for reducing boilerplate code.
2. Prioritize code simplicity, readability and performance optimization in every stage of development.
3. Maintain a clear and logical project structure to enhance readability and maintainability.
4. Follow consistent naming patterns for callbacks and functions

Azure Integration
- Use Azure-specific caching mechanisms
- Optimize for cloud performance

Refer to Dash, dash_core_components, dash_bootstrap_components, and dash_html_components documentation for best practices.

# Canadian STEM Graduate Analysis Dashboard - Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Core Functionality](#core-functionality)
3. [User Interface and Interactions](#user-interface-and-interactions)
4. [Examples and Use Cases](#examples-and-use-cases)
5. [Troubleshooting and FAQs](#troubleshooting-and-faqs)

## Project Overview

### Introduction
The Canadian STEM Graduate Analysis Dashboard is an interactive web application that visualizes and analyzes graduate education data across Canadian institutions. The dashboard focuses on:
- STEM (Science, Technology, Engineering, and Mathematics) graduates
- BHASE (Business, Humanities, Health, Arts, Social Science and Education) graduates
- Geographic distribution across Canada
- Educational levels and credential types
- Temporal trends

### Core Features
- Interactive choropleth map with graduate distribution
- Dynamic charts and visualizations
- Multi-dimensional filtering system
- Interactive pivot table for detailed analysis
- Data export capabilities

### Technical Architecture
- Frontend: Dash with Bootstrap components
- Data Processing: Pandas, GeoPandas
- Visualization: Plotly, Dash Leaflet
- Caching: Custom Azure-compatible system

### Installation Requirements
- Python 3.8 or higher
- Node.js and npm (for development)
- 2GB RAM minimum (4GB recommended)
- 1GB free disk space for caching

### Dependencies
```bash
# Core Libraries
dash>=2.0.0
dash-bootstrap-components>=1.0.0
dash-leaflet
dash-extensions
pandas
geopandas
plotly
numpy>=1.20.0
diskcache
colorlover

# Additional Requirements
dash-pivottable
orjson
dash-ag-grid
```

## Core Functionality

### Data Processing System
- Optimized filtering with caching
- Real-time data aggregation
- Multi-dimensional analysis
- Cross-filtering capabilities

### Geographic Visualization
- Custom map projections
- Interactive selection
- Dynamic color scaling
- Responsive tooltips

### Performance Optimization
- Two-layer caching system
- Memory and disk cache management
- Automatic cache cleanup
- Performance monitoring

### State Management
- Centralized application state
- Synchronized visualization updates
- Cross-component communication
- Error recovery mechanisms

## User Interface and Interactions

### Navigation
- Fixed-position navigation bar
- Two main tabs:
  1. Interactive Map and Charts
  2. Data Explorer
- Collapsible filter panel

### Filter System
- STEM/BHASE selection
- Academic year filters
- Geographic filters
- Educational level filters
- Institution filters

### Visualization Components
- Interactive choropleth map
- Bar charts for distributions
- Pivot table for detailed analysis
- Cross-filtering between components

### Data Export
- CSV export functionality
- Configurable pivot table exports
- Filtered data downloads
- Custom view exports

## Examples and Use Cases

### STEM vs BHASE Comparison
1. Initial Setup
2. Filter Application
3. Analysis Steps
4. Expected Outputs

### Regional Analysis
1. Geographic Selection
2. Data Exploration
3. Comparative Analysis
4. Export Options

### Temporal Analysis
1. Year Selection
2. Trend Analysis
3. Growth Patterns
4. Data Visualization

## Troubleshooting and FAQs

### Common Issues
1. Cache Performance
   - Symptoms
   - Solutions
   - Prevention

2. Data Loading
   - Error Scenarios
   - Recovery Steps
   - Best Practices

3. Visualization Problems
   - Identification
   - Resolution
   - Prevention

### Frequently Asked Questions
1. Filter Reset Process
2. Data Display Issues
3. Export Procedures
4. Analysis Techniques
5. Performance Optimization

## Data Reference

### Available Categories
1. STEM/BHASE
   - STEM
   - BHASE

2. Academic Years
   - 2019_2020
   - 2020_2021
   - 2021_2022

3. Credential Types
   - Associate degree
   - Attestation and other short program credentials
   - Certificate
   - Degree  (includes applied degree)
   - Diploma
   - General Equivalency Diploma/high school diploma
   - Other type of credential associated with a program

4. ISCED Levels
   - Bachelor’s or equivalent
   - Doctoral or equivalent
   - Master’s or equivalent
   - Not applicable
   - Post-secondary non-tertiary education
   - Short-cycle tertiary education
   - Upper secondary education


### Data Ranges
- Provincial totals: 0-843,294 graduates
- Institution-level: Variable by size
- CMA/CA level: Population-dependent