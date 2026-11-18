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
        mobileBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const isOpen = navLinks.classList.toggle('is-open');
            mobileBtn.classList.toggle('active', isOpen);
            mobileBtn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
            navLinks.style.display = isOpen ? 'flex' : 'none';
        });

        // Close when clicking outside
        document.addEventListener('click', (e) => {
            if (!mobileBtn.contains(e.target) && !navLinks.contains(e.target) && navLinks.classList.contains('is-open')) {
                navLinks.classList.remove('is-open');
                mobileBtn.classList.remove('active');
                mobileBtn.setAttribute('aria-expanded', 'false');
                navLinks.style.display = 'none';
            }
        });

        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && navLinks.classList.contains('is-open')) {
                navLinks.classList.remove('is-open');
                mobileBtn.classList.remove('active');
                mobileBtn.setAttribute('aria-expanded', 'false');
                navLinks.style.display = 'none';
            }
        });

        // Close when clicking any nav link
        navLinks.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                if (window.innerWidth <= 992 && navLinks.classList.contains('is-open')) {
                    navLinks.classList.remove('is-open');
                    mobileBtn.classList.remove('active');
                    mobileBtn.setAttribute('aria-expanded', 'false');
                    navLinks.style.display = 'none';
                }
            });
        });

        // Reset inline display when resizing to desktop
        window.addEventListener('resize', () => {
            if (window.innerWidth > 992) {
                navLinks.classList.remove('is-open');
                mobileBtn.classList.remove('active');
                mobileBtn.setAttribute('aria-expanded', 'false');
                navLinks.style.display = '';
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
