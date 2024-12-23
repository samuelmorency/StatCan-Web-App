// Client-side data processing functions

// Initialize client-side cache
const dataCache = {
    store: new Map(),
    maxSize: 50 * 1024 * 1024, // 50MB limit
    currentSize: 0,

    set(key, value) {
        const size = new Blob([JSON.stringify(value)]).size;
        if (size > this.maxSize) return false;
        
        while (this.currentSize + size > this.maxSize) {
            const oldestKey = this.store.keys().next().value;
            const oldSize = new Blob([JSON.stringify(this.store.get(oldestKey))]).size;
            this.store.delete(oldestKey);
            this.currentSize -= oldSize;
        }
        
        this.store.set(key, value);
        this.currentSize += size;
        return true;
    },

    get(key) {
        return this.store.get(key);
    }
};

// Client-side data filtering
function filterData(data, filters) {
    const cacheKey = JSON.stringify(filters);
    const cached = dataCache.get(cacheKey);
    if (cached) return cached;

    const filtered = data.filter(row => {
        return Object.entries(filters).every(([key, values]) => {
            if (!values || !values.length) return true;
            return values.includes(row[key]);
        });
    });

    dataCache.set(cacheKey, filtered);
    return filtered;
}

// Client-side aggregation
function aggregateData(data, groupBy, metric) {
    const cacheKey = `agg_${groupBy}_${metric}`;
    const cached = dataCache.get(cacheKey);
    if (cached) return cached;

    const result = data.reduce((acc, row) => {
        const key = row[groupBy];
        acc[key] = (acc[key] || 0) + row[metric];
        return acc;
    }, {});

    dataCache.set(cacheKey, result);
    return result;
}

// Client-side chart generation
function generateChart(data, type, options) {
    const cacheKey = `chart_${type}_${JSON.stringify(options)}`;
    const cached = dataCache.get(cacheKey);
    if (cached) return cached;

    // Chart generation logic here
    const chartData = {
        // Chart configuration
    };

    dataCache.set(cacheKey, chartData);
    return chartData;
}

// Export functions for use in Dash callbacks
window.clientProcessing = {
    filterData,
    aggregateData,
    generateChart,
    dataCache
};
