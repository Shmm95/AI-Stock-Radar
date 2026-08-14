# Workspace Policy

## Workspace A — Hardening (branch: workspace/a-hardening)
Scope: Zaten onaylanmış/aktif aşamanın sağlamlaştırılması — test yazımı,
optimizasyon, dokümantasyon/governance borcu kapatma, mevcut research
lineage'ının audit bulgularını düzeltme.
Bu workspace'in ürettiği değişiklikler, ana planlama sohbetinde (harici,
bu repoda değil) review edildikten sonra checkpoint branch'ine merge edilir.

## Workspace B — Forward Research (branch: workspace/b-forward-research)
Scope: Henüz onaylanmamış, ileri aşama deneysel çalışma — execution/slippage
stress, research baseline lock, broader-universe robustness, RSI replacement
research ve benzeri future-stage konular.
Bu workspace'in çıktısı HER ZAMAN "research candidate" statüsündedir.
Asla checkpoint branch'ine veya Workspace A'nın branch'ine doğrudan merge
edilmez. Ana planlama sohbetinde audit edilip açıkça onaylanmadan hiçbir
çıktısı official/approved sayılmaz.

## Protected Areas (Workspace B bunlara ASLA dokunamaz)
- src/backtest/portfolio_backtest_engine.py (deterministic engine)
- entry rules
- exit rules
- risk management logic
- execution order logic
- fees
- slippage
- frozen baseline strategy parametreleri
- config/research_baseline_lock_v1.json

Workspace B bu dosyalardan herhangi birini değiştirmesi gereken bir
duruma gelirse, değişikliği yapmadan durur ve ana planlama sohbetinde
onay ister.

## Merge Kuralı
Hiçbir worktree branch'i, ana planlama sohbetinde (bu repo dışında,
proje sahibiyle) açık onay alınmadan checkpoint branch'ine merge edilmez.
Bu kural her iki workspace için de CLAUDE.md içinde tekrar edilmelidir.
