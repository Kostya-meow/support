/**
 * Общий скрипт для управления навигацией на основе прав доступа
 */

// Получить права текущего пользователя
async function loadUserPermissions() {
    try {
        const response = await fetch('/api/user/permissions');
        if (!response.ok) {
            console.warn('Не удалось загрузить права пользователя');
            return [];
        }
        const data = await response.json();
        return data.permissions || [];
    } catch (error) {
        console.error('Ошибка загрузки прав:', error);
        return [];
    }
}

// Скрыть пункты меню, к которым нет доступа
function filterNavigationByPermissions(permissions) {
    const menuItems = {
        'tickets': 'a[href="/"]',
        'dashboard': 'a[href="/dashboard"]',
        'knowledge': 'a[href="/admin/knowledge"]',
        'simulator': 'a[href="/simulator"]',
        'admin': 'a[href="/admin/users"]'
    };

    // Скрываем все пункты, к которым нет доступа
    Object.entries(menuItems).forEach(([permission, selector]) => {
        const menuItem = document.querySelector(`.main-nav ${selector}`);
        if (menuItem) {
            if (permissions.includes(permission)) {
                menuItem.style.display = 'flex'; // Показываем
            } else {
                menuItem.style.display = 'none'; // Скрываем
            }
        }
    });
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', async () => {
    const permissions = await loadUserPermissions();
    filterNavigationByPermissions(permissions);
});

// Функция для сворачивания сайдбара (общая для всех страниц)
function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const toggle = document.querySelector('.sidebar-toggle');
    const icon = toggle?.querySelector('.material-icons');
    const isCollapsed = sidebar.classList.toggle('collapsed');
    
    // Меняем иконку Material Icons
    if (icon) {
        icon.textContent = isCollapsed ? 'chevron_right' : 'chevron_left';
    }
    
    localStorage.setItem('sidebarCollapsed', isCollapsed);
}

// Восстановление состояния сайдбара
document.addEventListener('DOMContentLoaded', () => {
    const collapsed = localStorage.getItem('sidebarCollapsed') === 'true';
    if (collapsed) {
        document.querySelector('.sidebar')?.classList.add('collapsed');
        const toggle = document.querySelector('.sidebar-toggle');
        const icon = toggle?.querySelector('.material-icons');
        if (icon) icon.textContent = 'chevron_right';
    }
});
