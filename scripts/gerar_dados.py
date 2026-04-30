"""
Gera dados sintéticos simulando aquisição de um cintilador
acoplado a um analisador multicanal (MCA).

Produz três ficheiros em ../data/:
  - calibracao.csv      : pares (canal, energia_keV) de fontes conhecidas
  - espectro_mca.csv    : espectro completo (canal, contagens) com Cs-137 + Co-60
  - resolucao_vs_E.csv  : FWHM medida vs energia (para ajuste polinomial)
"""

from pathlib import Path
import numpy as np

rng = np.random.default_rng(seed=42)

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)


def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# ---------------------------------------------------------------------------
# 1) Espectro MCA: Cs-137 (662 keV) + Co-60 (1173, 1332 keV) + fundo
# ---------------------------------------------------------------------------
N_CANAIS = 1024
GANHO = 2.0          # keV / canal
OFFSET = 5.0         # keV no canal 0
canais = np.arange(N_CANAIS)
energias = OFFSET + GANHO * canais

# fotopicos: (energia_keV, contagens_pico, resolucao_relativa)
picos = [
    (662.0,  1800, 0.07),   # Cs-137
    (1173.0, 1100, 0.055),  # Co-60
    (1332.0, 1000, 0.052),  # Co-60
]

espectro = np.zeros_like(energias)
for E0, A, res in picos:
    sigma = res * E0 / 2.355   # FWHM = 2.355 sigma
    espectro += gaussian(energias, A, E0, sigma)

# continuum de Compton (degraus suavizados antes de cada fotopico)
for E0, A, _ in picos:
    Ec = 2 * E0 ** 2 / (511 + 2 * E0)        # aresta de Compton
    degrau = 0.35 * A / (1 + np.exp((energias - Ec) / 8))
    espectro += degrau * (energias < E0)

# fundo exponencial + ruido de Poisson
espectro += 250 * np.exp(-energias / 400)
espectro = rng.poisson(np.clip(espectro, 0, None)).astype(float)

np.savetxt(
    DATA / "espectro_mca.csv",
    np.column_stack([canais, espectro]),
    fmt=["%d", "%d"],
    delimiter=",",
    header="canal,contagens",
    comments="",
)

# ---------------------------------------------------------------------------
# 2) Calibracao canal -> energia (5 fontes radioactivas conhecidas)
# ---------------------------------------------------------------------------
fontes = {
    "Am-241":  59.5,
    "Ba-133": 356.0,
    "Cs-137": 662.0,
    "Co-60a": 1173.0,
    "Co-60b": 1332.0,
}
energias_ref = np.array(list(fontes.values()))
canais_medidos = (energias_ref - OFFSET) / GANHO
canais_medidos += rng.normal(0, 1.5, size=canais_medidos.size)   # erro de leitura
sigma_canal = np.full_like(canais_medidos, 1.5)

with open(DATA / "calibracao.csv", "w") as f:
    f.write("fonte,canal,sigma_canal,energia_keV\n")
    for nome, E, c, sc in zip(fontes.keys(), energias_ref, canais_medidos, sigma_canal):
        f.write(f"{nome},{c:.2f},{sc:.2f},{E:.1f}\n")

# ---------------------------------------------------------------------------
# 3) Resolucao FWHM vs E  (depende ~ sqrt(E)  -> ajuste polinomial)
# ---------------------------------------------------------------------------
E_grid = np.array([60, 122, 356, 511, 662, 835, 1173, 1332, 1460])
fwhm = 2.5 * np.sqrt(E_grid) + 0.005 * E_grid
fwhm += rng.normal(0, 0.8, size=fwhm.size)
sigma_fwhm = np.full_like(fwhm, 0.8)

np.savetxt(
    DATA / "resolucao_vs_E.csv",
    np.column_stack([E_grid, fwhm, sigma_fwhm]),
    fmt=["%.1f", "%.3f", "%.3f"],
    delimiter=",",
    header="energia_keV,fwhm_keV,sigma_fwhm",
    comments="",
)

# ---------------------------------------------------------------------------
# 4) Espectro "mistério" para o projecto final: Eu-152
#    Calibração diferente: GANHO=1.5 keV/canal, OFFSET=10 keV
# ---------------------------------------------------------------------------
N_CANAIS_EU = 1024
GANHO_EU  = 1.5
OFFSET_EU = 10.0
canais_eu   = np.arange(N_CANAIS_EU)
energias_eu = OFFSET_EU + GANHO_EU * canais_eu

# Linhas gama do Eu-152 (energia em keV, intensidade relativa)
linhas_eu152 = [
    (121.78,  2800, 0.080),
    (244.70,   750, 0.070),
    (344.28,  2600, 0.062),
    (444.00,   280, 0.058),
    (778.90,  1300, 0.050),
    (964.06,  1450, 0.048),
    (1085.84, 1000, 0.046),
    (1112.08, 1350, 0.046),
    (1408.01, 2050, 0.042),
]

espectro_eu = np.zeros_like(energias_eu)
for E0, A, res in linhas_eu152:
    sigma = res * E0 / 2.355
    espectro_eu += gaussian(energias_eu, A, E0, sigma)

# Compton edges
for E0, A, _ in linhas_eu152:
    Ec = 2 * E0 ** 2 / (511 + 2 * E0)
    espectro_eu += 0.30 * A / (1 + np.exp((energias_eu - Ec) / 8)) * (energias_eu < E0)

# Background + Poisson
espectro_eu += 400 * np.exp(-energias_eu / 350)
espectro_eu = rng.poisson(np.clip(espectro_eu, 0, None)).astype(float)

np.savetxt(
    DATA / "espectro_misterio.csv",
    np.column_stack([canais_eu, espectro_eu]),
    fmt=["%d", "%d"],
    delimiter=",",
    header="canal,contagens",
    comments="",
)

print(f"Dados gerados em {DATA}/")
for p in sorted(DATA.glob("*.csv")):
    print(f"  - {p.name}")
