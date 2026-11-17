/**
 * RenalCare - Dashboard Analytics & Chart.js Integration
 * Adapts dynamically to light and dark theme palettes
 */

let outcomeChartInstance = null;
let modelChartInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    window.renderDashboardCharts();
});

window.renderDashboardCharts = function() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    
    const textColor = isDark ? '#EDEFEF' : '#1A201C';
    const mutedColor = isDark ? '#A2B0A8' : '#526058';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.06)';

    // 1. Screening Outcomes Chart
    const outcomeCanvas = document.getElementById('screeningOutcomeChart');
    if (outcomeCanvas) {
        const flaggedCount = parseInt(outcomeCanvas.getAttribute('data-flagged') || 0);
        const clearCount = parseInt(outcomeCanvas.getAttribute('data-clear') || 0);

        if (outcomeChartInstance) {
            outcomeChartInstance.destroy();
        }

        const total = flaggedCount + clearCount;
        const dataValues = total === 0 ? [1] : [flaggedCount, clearCount];
        const bgColors = total === 0 ? 
            [isDark ? '#2D3A35' : '#E5E8E5'] : 
            [isDark ? '#F87171' : '#DC2626', isDark ? '#88B39F' : '#174D3A'];

        outcomeChartInstance = new Chart(outcomeCanvas, {
            type: 'doughnut',
            data: {
                labels: total === 0 ? ['No Data Yet'] : ['Flagged (CKD Indicators)', 'Non-Flagged (Baseline)'],
                datasets: [{
                    data: dataValues,
                    backgroundColor: bgColors,
                    borderWidth: 2,
                    borderColor: isDark ? '#202A26' : '#FFFFFF'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '72%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: textColor,
                            font: { family: "'Plus Jakarta Sans', sans-serif", size: 12 },
                            padding: 16
                        }
                    },
                    tooltip: {
                        enabled: total > 0
                    }
                }
            }
        });
    }

    // 2. Model Performance Comparison Chart
    const modelCanvas = document.getElementById('modelComparisonChart');
    if (modelCanvas) {
        if (modelChartInstance) {
            modelChartInstance.destroy();
        }

        // Metrics from held-out test evaluation on UCI dataset
        // Decision Tree: Acc 96.2%, Recall 94.0%, Spec 100.0%, Prec 100.0%, F1 96.9%
        // Random Forest: Acc 100.0%, Recall 100.0%, Spec 100.0%, Prec 100.0%, F1 100.0%
        modelChartInstance = new Chart(modelCanvas, {
            type: 'bar',
            data: {
                labels: ['Accuracy', 'Sensitivity (Recall)', 'Specificity', 'Precision', 'F1-Score'],
                datasets: [
                    {
                        label: 'Random Forest (Champion)',
                        data: [100.0, 100.0, 100.0, 100.0, 100.0],
                        backgroundColor: isDark ? '#88B39F' : '#174D3A',
                        borderRadius: 4
                    },
                    {
                        label: 'Decision Tree',
                        data: [96.2, 94.0, 100.0, 100.0, 96.9],
                        backgroundColor: isDark ? '#6E9382' : '#7A9A8B',
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 105,
                        grid: { color: gridColor },
                        ticks: {
                            color: mutedColor,
                            callback: function(value) { return value + '%'; }
                        }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: textColor, font: { weight: '600' } }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: textColor,
                            font: { family: "'Plus Jakarta Sans', sans-serif", size: 12 },
                            padding: 14
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + context.parsed.y + '%';
                            }
                        }
                    }
                }
            }
        });
    }
};
