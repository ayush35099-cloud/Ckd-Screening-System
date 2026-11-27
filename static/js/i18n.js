/**
 * RenalCare - Multilingual Translation, Accessibility (Easy Mode) & Speech Synthesis Engine
 * Supports 8 languages: English, Hindi, Bengali, Marathi, Telugu, Tamil, Urdu (RTL), Gujarati
 */

(function () {
    'use strict';

    const STORAGE_KEY_LANG = 'renalcare_lang';
    const STORAGE_KEY_FIRST_VISIT = 'renalcare_visited';
    const STORAGE_KEY_EASY_MODE = 'renalcare_easy_mode';

    // Voice codes for SpeechSynthesis
    const LANG_VOICE_MAP = {
        'en': ['en-IN', 'en-GB', 'en-US', 'en'],
        'hi': ['hi-IN', 'hi'],
        'bn': ['bn-IN', 'bn-BD', 'bn'],
        'mr': ['mr-IN', 'mr'],
        'te': ['te-IN', 'te'],
        'ta': ['ta-IN', 'ta-LK', 'ta'],
        'ur': ['ur-PK', 'ur-IN', 'ur'],
        'gu': ['gu-IN', 'gu']
    };

    let currentLang = 'en';
    let isEasyMode = false;
    let activeUtterance = null;

    /**
     * Get initial language preference
     */
    function getStoredLanguage() {
        const urlParams = new URLSearchParams(window.location.search);
        const urlLang = urlParams.get('lang');
        if (urlLang && window.RENALCARE_LANG_META && window.RENALCARE_LANG_META[urlLang]) {
            return urlLang;
        }

        const stored = localStorage.getItem(STORAGE_KEY_LANG);
        if (stored && window.RENALCARE_LANG_META && window.RENALCARE_LANG_META[stored]) {
            return stored;
        }

        // Check cookie
        const cookieMatch = document.cookie.match(new RegExp('(^|;\\s*)renalcare_lang=([^;]+)'));
        if (cookieMatch && window.RENALCARE_LANG_META && window.RENALCARE_LANG_META[cookieMatch[2]]) {
            return cookieMatch[2];
        }

        return null; // Signals first-time visitor
    }

    /**
     * Set language and persist
     */
    function setLanguage(langCode, updateUrl = false) {
        if (!window.RENALCARE_LANG_META || !window.RENALCARE_LANG_META[langCode]) {
            langCode = 'en';
        }

        currentLang = langCode;
        localStorage.setItem(STORAGE_KEY_LANG, langCode);
        localStorage.setItem(STORAGE_KEY_FIRST_VISIT, 'true');
        document.cookie = `renalcare_lang=${langCode}; path=/; max-age=31536000; SameSite=Lax`;

        // Update HTML attributes
        const meta = window.RENALCARE_LANG_META[langCode] || { dir: 'ltr' };
        document.documentElement.setAttribute('lang', langCode);
        document.documentElement.setAttribute('dir', meta.dir || 'ltr');

        if (meta.dir === 'rtl') {
            document.body.classList.add('rtl-layout');
        } else {
            document.body.classList.remove('rtl-layout');
        }

        // Apply translations across all marked elements
        translateDOM(langCode);

        // Update active class in dropdown & modal
        updateLanguageSelectors(langCode);

        // Update PDF report download links to include current language
        updatePdfReportLinks(langCode);

        // Optional URL parameter update
        if (updateUrl && window.history && window.history.replaceState) {
            const url = new URL(window.location.href);
            url.searchParams.set('lang', langCode);
            window.history.replaceState({}, '', url.toString());
        }

        // Re-render Lucide icons if present
        if (window.lucide && typeof window.lucide.createIcons === 'function') {
            window.lucide.createIcons();
        }

        // Trigger custom event for other components
        document.dispatchEvent(new CustomEvent('renalcare:languageChanged', { detail: { lang: langCode } }));
    }

    /**
     * Translate DOM nodes
     */
    function translateDOM(langCode) {
        const translations = (window.RENALCARE_TRANSLATIONS && window.RENALCARE_TRANSLATIONS[langCode]) || {};
        const fallback = (window.RENALCARE_TRANSLATIONS && window.RENALCARE_TRANSLATIONS['en']) || {};

        // Translate text contents
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            if (!key) return;

            let text = translations[key] || fallback[key];
            if (text !== undefined) {
                // If element has child icons or tags, replace only text nodes or inner HTML safely
                if (el.hasAttribute('data-i18n-html')) {
                    el.innerHTML = text;
                } else if (el.querySelector('i, svg, span.badge, span.ref-pill')) {
                    // Find direct text node or span
                    const textSpan = el.querySelector('.i18n-text');
                    if (textSpan) {
                        textSpan.textContent = text;
                    } else {
                        // Preserve icon and replace text
                        const icon = el.querySelector('i, svg');
                        el.textContent = ' ' + text;
                        if (icon) el.prepend(icon);
                    }
                } else {
                    el.textContent = text;
                }
            }
        });

        // Translate input placeholders
        document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
            const key = el.getAttribute('data-i18n-placeholder');
            const text = translations[key] || fallback[key];
            if (text !== undefined) {
                el.setAttribute('placeholder', text);
            }
        });

        // Translate titles & aria-labels
        document.querySelectorAll('[data-i18n-title]').forEach(el => {
            const key = el.getAttribute('data-i18n-title');
            const text = translations[key] || fallback[key];
            if (text !== undefined) {
                el.setAttribute('title', text);
                el.setAttribute('aria-label', text);
            }
        });
    }

    /**
     * Update active state of language dropdown and buttons
     */
    function updateLanguageSelectors(langCode) {
        // Dropdown buttons
        document.querySelectorAll('.lang-dropdown-item').forEach(item => {
            if (item.getAttribute('data-lang') === langCode) {
                item.classList.add('active');
                item.setAttribute('aria-selected', 'true');
            } else {
                item.classList.remove('active');
                item.setAttribute('aria-selected', 'false');
            }
        });

        // Current active label in nav
        const currentLangLabel = document.getElementById('currentLangLabel');
        if (currentLangLabel && window.RENALCARE_LANG_META && window.RENALCARE_LANG_META[langCode]) {
            currentLangLabel.textContent = window.RENALCARE_LANG_META[langCode].native;
        }
    }

    /**
     * Update PDF download links to pass ?lang=...
     */
    function updatePdfReportLinks(langCode) {
        document.querySelectorAll('a[href*="/report/"]').forEach(link => {
            const href = link.getAttribute('href');
            if (href && !href.startsWith('javascript:')) {
                const url = new URL(href, window.location.origin);
                url.searchParams.set('lang', langCode);
                link.setAttribute('href', url.pathname + url.search);
            }
        });
    }

    /**
     * First-Visit Welcome Modal
     */
    function initWelcomeModal() {
        const welcomeModal = document.getElementById('welcomeLanguageModal');
        if (!welcomeModal) return;

        const storedLang = getStoredLanguage();
        if (!storedLang) {
            // First time visitor: Show modal
            welcomeModal.style.display = 'flex';
            document.body.style.overflow = 'hidden';
        } else {
            welcomeModal.style.display = 'none';
            setLanguage(storedLang, false);
        }

        // Language cards in modal
        const cards = welcomeModal.querySelectorAll('.welcome-lang-card');
        let selectedInModal = storedLang || 'en';

        cards.forEach(card => {
            card.addEventListener('click', () => {
                cards.forEach(c => c.classList.remove('active'));
                card.classList.add('active');
                selectedInModal = card.getAttribute('data-lang');
                
                // Live preview translation inside modal buttons
                const btnContinue = document.getElementById('btnWelcomeContinue');
                if (btnContinue && window.RENALCARE_TRANSLATIONS && window.RENALCARE_TRANSLATIONS[selectedInModal]) {
                    btnContinue.textContent = window.RENALCARE_TRANSLATIONS[selectedInModal]['welcome_continue'] || 'Continue to RenalCare';
                }
            });
        });

        const btnContinue = document.getElementById('btnWelcomeContinue');
        if (btnContinue) {
            btnContinue.addEventListener('click', () => {
                setLanguage(selectedInModal, true);
                welcomeModal.style.opacity = '0';
                setTimeout(() => {
                    welcomeModal.style.display = 'none';
                    document.body.style.overflow = '';
                }, 300);
            });
        }
    }

    /**
     * Navbar Language Dropdown
     */
    function initNavbarDropdown() {
        const toggleBtn = document.getElementById('navLangToggle');
        const dropdownMenu = document.getElementById('navLangDropdown');
        if (!toggleBtn || !dropdownMenu) return;

        toggleBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const isOpen = dropdownMenu.classList.contains('show');
            if (isOpen) {
                dropdownMenu.classList.remove('show');
                toggleBtn.setAttribute('aria-expanded', 'false');
            } else {
                dropdownMenu.classList.add('show');
                toggleBtn.setAttribute('aria-expanded', 'true');
            }
        });

        document.addEventListener('click', (e) => {
            if (!toggleBtn.contains(e.target) && !dropdownMenu.contains(e.target)) {
                dropdownMenu.classList.remove('show');
                toggleBtn.setAttribute('aria-expanded', 'false');
            }
        });

        dropdownMenu.querySelectorAll('.lang-dropdown-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const lang = item.getAttribute('data-lang');
                if (lang) {
                    setLanguage(lang, true);
                    dropdownMenu.classList.remove('show');
                    toggleBtn.setAttribute('aria-expanded', 'false');
                }
            });
        });
    }

    /**
     * Easy Mode (Rural Accessibility)
     */
    function initEasyMode() {
        const easyModeToggle = document.getElementById('btnEasyModeToggle');
        const storedEasy = localStorage.getItem(STORAGE_KEY_EASY_MODE);
        isEasyMode = storedEasy === 'true';

        applyEasyMode(isEasyMode);

        if (easyModeToggle) {
            easyModeToggle.addEventListener('click', () => {
                isEasyMode = !isEasyMode;
                localStorage.setItem(STORAGE_KEY_EASY_MODE, isEasyMode ? 'true' : 'false');
                applyEasyMode(isEasyMode);
            });
        }
    }

    function applyEasyMode(active) {
        const easyModeToggle = document.getElementById('btnEasyModeToggle');
        const banner = document.getElementById('easyModeBanner');

        if (active) {
            document.documentElement.setAttribute('data-easy-mode', 'true');
            if (easyModeToggle) {
                easyModeToggle.classList.add('active');
                easyModeToggle.setAttribute('aria-pressed', 'true');
            }
            if (banner) banner.style.display = 'block';
            document.querySelectorAll('.easy-mode-helper').forEach(h => h.style.display = 'block');
        } else {
            document.documentElement.removeAttribute('data-easy-mode');
            if (easyModeToggle) {
                easyModeToggle.classList.remove('active');
                easyModeToggle.setAttribute('aria-pressed', 'false');
            }
            if (banner) banner.style.display = 'none';
            document.querySelectorAll('.easy-mode-helper').forEach(h => h.style.display = 'none');
        }
    }

    /**
     * SpeechSynthesis (Voice Instructions)
     */
    function initSpeechSynthesis() {
        if (!('speechSynthesis' in window)) {
            console.log('Web Speech API is not supported in this browser.');
            return;
        }

        document.querySelectorAll('.btn-tts').forEach(btn => {
            btn.addEventListener('click', () => {
                if (window.speechSynthesis.speaking) {
                    window.speechSynthesis.cancel();
                    btn.classList.remove('speaking');
                    const textSpan = btn.querySelector('.tts-label');
                    if (textSpan) {
                        const translations = (window.RENALCARE_TRANSLATIONS && window.RENALCARE_TRANSLATIONS[currentLang]) || {};
                        textSpan.textContent = translations['hero_listen'] || 'Listen to Instructions';
                    }
                    const icon = btn.querySelector('i');
                    if (icon) {
                        icon.setAttribute('data-lucide', 'volume-2');
                        if (window.lucide) window.lucide.createIcons();
                    }
                    return;
                }

                // Determine text to speak
                let textToSpeak = '';
                const targetSelector = btn.getAttribute('data-tts-target');
                if (targetSelector) {
                    const targetEl = document.querySelector(targetSelector);
                    if (targetEl) textToSpeak = targetEl.innerText || targetEl.textContent;
                } else if (btn.getAttribute('data-tts-text')) {
                    textToSpeak = btn.getAttribute('data-tts-text');
                } else {
                    // Default to hero intro or parent card
                    const parentCard = btn.closest('.hero-content, .clinical-fieldset, .result-card-main, .form-hero-header');
                    if (parentCard) textToSpeak = parentCard.innerText || parentCard.textContent;
                }

                if (!textToSpeak.trim()) return;

                speakText(textToSpeak, currentLang, btn);
            });
        });
    }

    function speakText(text, langCode, buttonEl) {
        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.95; // Slightly slower for clear clinical understanding
        utterance.pitch = 1.0;

        // Try to match appropriate voice
        const voices = window.speechSynthesis.getVoices();
        const preferredTags = LANG_VOICE_MAP[langCode] || ['en-IN', 'en'];

        let selectedVoice = null;
        for (const tag of preferredTags) {
            selectedVoice = voices.find(v => v.lang.toLowerCase().replace('_', '-').startsWith(tag.toLowerCase()));
            if (selectedVoice) break;
        }

        if (selectedVoice) {
            utterance.voice = selectedVoice;
            utterance.lang = selectedVoice.lang;
        } else {
            utterance.lang = preferredTags[0];
            // If non-English and no voice found, notify user gracefully
            if (langCode !== 'en' && voices.length > 0) {
                const hasAnyLangVoice = voices.some(v => v.lang.toLowerCase().startsWith(langCode));
                if (!hasAnyLangVoice) {
                    showVoiceAlert(langCode);
                }
            }
        }

        utterance.onstart = () => {
            if (buttonEl) {
                buttonEl.classList.add('speaking');
                const textSpan = buttonEl.querySelector('.tts-label');
                if (textSpan) {
                    const translations = (window.RENALCARE_TRANSLATIONS && window.RENALCARE_TRANSLATIONS[langCode]) || {};
                    textSpan.textContent = translations['hero_stop'] || 'Stop Listening';
                }
                const icon = buttonEl.querySelector('i');
                if (icon) {
                    icon.setAttribute('data-lucide', 'volume-x');
                    if (window.lucide) window.lucide.createIcons();
                }
            }
        };

        utterance.onend = utterance.onerror = () => {
            if (buttonEl) {
                buttonEl.classList.remove('speaking');
                const textSpan = buttonEl.querySelector('.tts-label');
                if (textSpan) {
                    const translations = (window.RENALCARE_TRANSLATIONS && window.RENALCARE_TRANSLATIONS[langCode]) || {};
                    textSpan.textContent = translations['hero_listen'] || 'Listen to Instructions';
                }
                const icon = buttonEl.querySelector('i');
                if (icon) {
                    icon.setAttribute('data-lucide', 'volume-2');
                    if (window.lucide) window.lucide.createIcons();
                }
            }
        };

        window.speechSynthesis.speak(utterance);
    }

    function showVoiceAlert(langCode) {
        const meta = window.RENALCARE_LANG_META[langCode] || { name: langCode };
        const msg = `${meta.name} voice is not installed on this browser/device. Audio will use the system default voice.`;
        
        let alertBox = document.getElementById('voiceNoticeBanner');
        if (!alertBox) {
            alertBox = document.createElement('div');
            alertBox.id = 'voiceNoticeBanner';
            alertBox.style.cssText = 'position: fixed; bottom: 20px; right: 20px; background: var(--bg-surface); border: 1px solid var(--border-strong); padding: 12px 18px; border-radius: 8px; box-shadow: var(--shadow-md); z-index: 9999; font-size: 0.85rem; max-width: 320px;';
            document.body.appendChild(alertBox);
        }
        alertBox.innerHTML = `<strong>Audio Notification:</strong><p style="margin-top: 4px; color: var(--text-muted);">${msg}</p>`;
        alertBox.style.display = 'block';
        setTimeout(() => { alertBox.style.display = 'none'; }, 4500);
    }

    // Ensure voices are loaded (Chrome loads voices asynchronously)
    if ('speechSynthesis' in window && window.speechSynthesis.onvoiceschanged !== undefined) {
        window.speechSynthesis.onvoiceschanged = () => {
            window.speechSynthesis.getVoices();
        };
    }

    // DOM Ready Initialization
    document.addEventListener('DOMContentLoaded', () => {
        initWelcomeModal();
        initNavbarDropdown();
        initEasyMode();
        initSpeechSynthesis();

        // Apply initially stored language or fallback to English
        const lang = getStoredLanguage() || 'en';
        setLanguage(lang, false);
    });

    // Expose API globally
    window.RenalCareI18n = {
        setLanguage: setLanguage,
        getLanguage: () => currentLang,
        toggleEasyMode: () => {
            isEasyMode = !isEasyMode;
            localStorage.setItem(STORAGE_KEY_EASY_MODE, isEasyMode ? 'true' : 'false');
            applyEasyMode(isEasyMode);
        },
        speak: speakText
    };

})();
