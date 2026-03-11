// GRB_Technology Virtual Try-On — client-side JS

function setupImagePreview(inputId, previewId, placeholderId, statusId) {
  const input       = document.getElementById(inputId);
  const preview     = document.getElementById(previewId);
  const placeholder = document.getElementById(placeholderId);
  const status      = document.getElementById(statusId);
  if (!input) return;

  input.addEventListener('change', function () {
    const file = this.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => {
      preview.src = e.target.result;
      preview.style.display = 'block';
      placeholder.style.display = 'none';
      status.textContent = '✅ ' + file.name;
    };
    reader.readAsDataURL(file);
  });
}

setupImagePreview('person-input', 'person-preview', 'person-placeholder', 'person-status');
setupImagePreview('cloth-input',  'cloth-preview',  'cloth-placeholder',  'cloth-status');

// Disable submit button and show loading text while processing
const form = document.getElementById('tryon-form');
const btn  = document.getElementById('submit-btn');
if (form && btn) {
  form.addEventListener('submit', () => {
    btn.disabled = true;
    btn.textContent = '⏳ Processing… (~30s)';
  });
}
