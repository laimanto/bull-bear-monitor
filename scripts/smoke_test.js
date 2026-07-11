const fs = require('fs');
const { JSDOM } = require('jsdom');
const html = fs.readFileSync(process.argv[2] || 'D:/Backup D/Weekly/USB drive/Invest/AI invest/Bolinger/Bollinger_Dashboard.html', 'utf8');

const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  pretendToBeVisual: true,
  beforeParse(window) {
    window.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
      get: (t, p) => typeof p === 'string' ? function () { return 0; } : undefined,
      set: () => true,
    });
    window.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
    window.addEventListener('error', e => { console.log('PAGE ERROR:', e.message); });
  },
});
const d = dom.window.document;
setTimeout(() => {
  const tabs = [...d.querySelectorAll('.tab')].map(t => t.textContent);
  console.log('tabs:', tabs.join(', '));
  console.log('panels:', d.querySelectorAll('.panel').length);
  for (const tab of d.querySelectorAll('.tab')) {
    tab.click();
    const id = tab.dataset.target;
    const panel = d.getElementById('panel-' + id);
    const active = panel && panel.classList.contains('active');
    const rows = panel ? panel.querySelectorAll('tbody tr').length : 0;
    console.log(id, 'active:', active, 'trade rows:', rows);
  }
  console.log('tiles on NVDA panel:', d.querySelectorAll('#panel-NVDA .tile').length);
  console.log('SMOKE TEST DONE');
  dom.window.close();
}, 300);
