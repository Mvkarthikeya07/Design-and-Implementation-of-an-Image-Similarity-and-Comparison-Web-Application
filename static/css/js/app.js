// static/js/app.js - simple client helpers (future UX enhancements)
document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('uploadForm');
  if (!form) return;
  form.addEventListener('submit', function () {
    // show simple feedback by disabling button
    const btn = form.querySelector('button[type="submit"]');
    if (btn) {
      btn.disabled = true;
      btn.innerText = 'Comparing...';
    }
  });
});
