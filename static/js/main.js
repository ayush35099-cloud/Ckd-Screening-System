/**
 * RenalCare - Core UI JavaScript
 * Theme management, navigation, and global utilities
 */

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initMobileNav();
    initIcons();
});

// 1. Dark/Light Theme Controller
function initTheme() {
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    const savedTheme = localStorage.getItem('renalcare-theme') || 'light';
    
    // Apply saved theme
    document.documentElement.setAttribute('data-theme', savedTheme);

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('renalcare-theme', newTheme);

            // Re-render charts if present on page
            if (window.renderDashboardCharts) {
                window.renderDashboardCharts();
            }
        });
    }
}

// 2. Mobile Navigation Toggle
function initMobileNav() {
    const mobileBtn = document.getElementById('mobileMenuBtn');
    const navLinks = document.getElementById('navLinks');

    if (mobileBtn && navLinks) {
        mobileBtn.addEventListener('click', () => {
            const isVisible = navLinks.style.display === 'flex';
            navLinks.style.display = isVisible ? 'none' : 'flex';
            if (!isVisible) {
                navLinks.style.flexDirection = 'column';
                navLinks.style.position = 'absolute';
                navLinks.style.top = '100%';
                navLinks.style.left = '0';
                navLinks.style.right = '0';
                navLinks.style.backgroundColor = 'var(--bg-surface)';
                navLinks.style.padding = '1.5rem';
                navLinks.style.borderBottom = '1px solid var(--border-subtle)';
                navLinks.style.boxShadow = 'var(--shadow-md)';
            }
        });
    }
}

// 3. Lucide Icons Initializer
function initIcons() {
    if (window.lucide) {
        window.lucide.createIcons();
    }
}
