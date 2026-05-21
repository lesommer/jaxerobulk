# Bulk Formulae in AeroBulk / JaxeroBulk

## 1. General Formulation

Aerodynamic bulk formulae compute the turbulent fluxes of momentum, sensible heat,
and latent heat (evaporation) at the air-sea interface. Given surface and atmospheric
state variables, they determine transfer coefficients and fluxes through Monin-Obukhov
similarity theory (MOST).

### Core Equations

The four primary fluxes are:

| Flux | Symbol | Formula |
|------|--------|---------|
| Wind stress | τ | ρ_air · C_D · U_z · U_B |
| Sensible heat | Q_H | ρ_air · C_H · C_P · [θ_z − T_s] · U_B |
| Evaporation | E | ρ_air · C_E · [q_z − q_s] · U_B |
| Latent heat | Q_L | −L_v · E |

where:

- **ρ_air**: air density at height z [kg/m³]
- **C_D, C_H, C_E**: bulk transfer coefficients for momentum, heat, and moisture
- **U_z**: scalar wind speed at height z [m/s]
- **U_B**: bulk wind speed (includes gustiness in some schemes) [m/s]
- **C_P**: specific heat of moist air [J/K/kg]
- **θ_z**: potential temperature at height z [K]
- **T_s**: sea surface skin temperature [K]
- **q_z**: specific humidity at height z [kg/kg]
- **q_s**: saturation specific humidity at the sea surface [kg/kg]
- **L_v**: latent heat of vaporization [J/kg]

### Approximations

- Potential temperature: θ_z ≈ T_z + γ·z, where γ is the dry adiabatic lapse rate
- Surface saturation humidity: q_s ≈ 0.98 · q_sat(T_s, P_0), the 0.98 factor accounts
  for the salinity reduction of saturation vapor pressure over seawater

### Wind Stress Decomposition

The wind stress vector is decomposed as:
- τ_x = (|τ| / |U|) · U_z
- τ_y = (|τ| / |U|) · V_z

### Air Density

ρ_air = P / (R_dry · T · (1 + 0.608 · q)), floored at 0.8 kg/m³

### Latent Heat of Vaporization

L_v = (2.501 − 0.00237 · (SST − 273.15)) × 10⁶ [J/kg]

---

## 2. Monin-Obukhov Similarity Theory (MOST)

All schemes share the MOST framework. The dimensionless stability parameter is:

ζ = z / L

where L is the Obukhov length:

1/L = κ · g · (θ* + 0.61 · T · q*) / (u*² · θ_v)

Profile functions relate mean gradients to turbulent scales:

- Wind: U(z) = (u*/κ) · [ln(z/z₀) − ψ_m(ζ)]
- Temperature: θ(z) − θ_s = (θ*/κ) · [ln(z/z₀t) − ψ_h(ζ)]
- Humidity: q(z) − q_s = (q*/κ) · [ln(z/z₀q) − ψ_h(ζ)]

where:
- κ = 0.4 is the von Kármán constant
- z₀, z₀t, z₀q are roughness lengths for momentum, heat, and moisture
- ψ_m, ψ_h are integrated stability correction functions
- u*, θ*, q* are turbulent scaling parameters

### Inverse Obukhov Length

1/L = g · κ · (θ_v* ) / (u*² · θ_v)

where θ_v* = θ* + 0.61 · T · q* is the virtual potential temperature scale.

### Bulk Richardson Number

Ri_b = g · Δθ_v · z / (θ_v · U²)

Used in some schemes (ECMWF) to initialize the stability estimate.

---

## 3. The Iteration Procedure

All bulk algorithms iterate to find consistent values of the transfer coefficients
given the stability state. The general procedure is:

1. **First guess**: Estimate u*, θ*, q*, z₀ from neutral conditions or Richardson number
2. **Iteration loop** (typically 4-6 iterations):
   a. Compute 1/L from current u*, θ*, q*
   b. Compute gustiness U_g (schemes that include it)
   c. Update bulk wind: U_B = √(U² + U_g²)
   d. Update roughness z₀ (Charnock + smooth term)
   e. Update scalar roughness z₀t, z₀q
   f. Compute stability parameters ζ_u, ζ_t
   g. Update u*, θ*, q* from profile equations with ψ_m, ψ_h
   h. Adjust T, q from z_t to z_u if heights differ
   i. [If cool-skin]: Update T_s, q_s for cool-skin correction
   j. [If warm-layer]: Update T_s, q_s for warm-layer correction
3. **Compute transfer coefficients**:
   - C_D = (u*/U_B)²
   - C_H = (u*/U_B) · θ*/Δθ
   - C_E = (u*/U_B) · q*/Δq

---

## 4. Roughness Length Parameterizations

### Momentum Roughness (z₀)

All ocean schemes use the Charnock relation plus a smooth-flow term:

z₀ = α · u*²/g + β · ν_a/u*

where α is the Charnock parameter (varies by scheme and wind speed), β is a
smooth-flow coefficient, and ν_a is the kinematic viscosity of air.

### Scalar Roughness (z₀t, z₀q)

Each scheme uses different formulations (see per-scheme details below).

---

## 5. Algorithm-Specific Details

### 5.1 COARE 3.0

**Reference**: Fairall et al. (2003), J. Climate

**Charnock parameter** (piecewise):
- U < 10 m/s: α = 0.011
- 10 ≤ U ≤ 18 m/s: linear interpolation from 0.011 to 0.018
- U > 18 m/s: α = 0.018

**Scalar roughness**:
- z₀t = z₀q = min(1.1×10⁻⁴, 5.5×10⁻⁵ · Re_r^(−0.6))
- where Re_r = z₀ · u*/ν_a is the roughness Reynolds number

**Gustiness**:
- U_g² = β₀² · u*² · max(−z_i/(κ·L), 0)^(2/3)
- β₀ = 1.25, z_i = 600 m (ABL height)

**Stability functions**:
- **Unstable** (ζ < 0): Blended Kansas + convective form
  - Kansas: Paulson (1970), using (1 − 15ζ)^0.25
  - Convective: (1 − 10.15ζ)^0.333
  - Blending weight: f = ζ²/(1 + ζ²)
- **Stable** (ζ ≥ 0): Beljaars & Holtslag (1991) with exponential decay

**Cool-skin/warm-layer**: Supported (Fairall et al. 1996 cool-skin, Fairall et al. 2019 warm-layer)

**Warm-layer depth**: Prognostic, max 20 m

**Cool-skin solar absorption**: fr = max(0.137 + 11δ − 6.6×10⁻⁵/δ · (1−exp(−δ/8×10⁻⁴)), 0.01)

---

### 5.2 COARE 3.6

**Reference**: Edson et al. (2013), Fairall et al. (2016)

**Charnock parameter** (continuous linear in neutral wind):
- α = max(min(0.0017 · U_N10 − 0.005, 0.028), 0)

**Scalar roughness**:
- z₀t = z₀q = min(1.6×10⁻⁴, 5.8×10⁻⁵ · Re_r^(−0.72))
- Note: exponent 0.72 (vs 0.6 in COARE 3.0), different constants

**Gustiness**: Same formulation as COARE 3.0 but β₀ = 1.2 (reduced from 1.25)

**Stability functions**: Same as COARE 3.0

**Cool-skin/warm-layer**: Supported, same as COARE 3.0

**Additional**: Includes a wave-dependent Charnock parameter option (COARE 3.5)

---

### 5.3 NCAR (Large & Yeager)

**Reference**: Large & Yeager (2004/2008), Large & Pond (1981)

**Neutral transfer coefficients as functions of wind speed**:
- C_DN10 = 10⁻³ · (2.7/U₁₀ + 0.142 + U₁₀/13.09 − 3.148×10⁻¹⁰ · U₁₀⁶) for U₁₀ < 33 m/s
- C_DN10 capped at 2.34×10⁻³ for U₁₀ ≥ 33 m/s
- C_HN10 = 10⁻³ · √(C_DN) · (18·s + 32.7·(1−s)), where s=1 (stable) or 0 (unstable)
- C_EN10 = 10⁻³ · 34.6 · √(C_DN)

**Stability functions**:
- **Unstable** (ζ < 0): Paulson (1970)
- **Stable** (ζ ≥ 0): ψ = −5·ζ (simple linear)

**Gustiness**: None

**Cool-skin/warm-layer**: Not supported

**Iteration**: Updates C_D from C_DN using stability corrections (Large & Yeager 2004 Eqs. 10a-10c)

**Minimum wind speed**: 0.5 m/s floor on U_B

---

### 5.4 ECMWF (IFS Cy40r1)

**Reference**: IFS Documentation Cy40r1, Zeng & Beljaars (2005)

**Charnock parameter**: Fixed α = 0.018

**Smooth-flow roughness** (separate for each variable):
- z₀ = 0.11 · ν/u* + α · u*²/g  (momentum)
- z₀t = 0.40 · ν/u* (heat)
- z₀q = 0.62 · ν/u* (moisture)

**Gustiness**:
- Same formulation as COARE but β₀ = 1.0, z_i = 1000 m

**Stability functions**:
- **Unstable** (ζ < 0): Paulson (1970) — same as NCAR
- **Stable** (ζ ≥ 0): Beljaars-Holtslag (1991) form, with stability parameter capped

**Transfer coefficients** (profile-based, not from u*):
- C_D = κ² / (F_m · F_m)
- C_H = κ² / (F_m · F_h)
- C_E = κ² / (F_m · F_q)
- where F_m = ln(z_u/z₀) − ψ_m(ζ_u) + ψ_m(z₀/L), etc.

**Stability initialization**: Uses bulk Richardson number rather than direct 1/L

**ζ capping**: ζ limited to [−50, 5] for numerical stability

**Cool-skin**: Fairall et al. 1996 with solar absorption coefficient 0.065 (vs 0.137 in COARE)

**Warm-layer**: Zeng & Beljaars (2005) — fixed depth rd₀ = 3 m, semi-implicit iteration (10 iterations), includes Langmuir turbulence option and Takaya et al. (2010) stability function

---

### 5.5 Andreas (Sea Ice)

**Reference**: Andreas et al. (2015)

**Friction velocity** (direct polynomial):
- u* = 0.239 + 0.0433 · (U_N10 − 8.271 + √(0.12·(U_N10−8.271)² + 0.181))

**Scalar roughness**: Liu-Katsaros-Businger (1979) — lookup table with 8 ranges
of roughness Reynolds number

**Stability functions**:
- **Unstable** (ζ < 0): Paulson (1970)
- **Stable** (ζ ≥ 0): Grachev et al. (2007) — complex cubic form from SHEBA data

**Safety measures**:
- Maximum bulk Richardson number: 0.15
- Minimum C_H, C_E: 0.35×10⁻³
- When Ri_b exceeds maximum, u* forced consistent with minimum transfer coefficients

**Gustiness**: None

**Cool-skin/warm-layer**: Not supported

---

## 6. Cool-Skin Parameterization

The cool-skin correction accounts for the temperature difference between the ocean
skin (top few millimeters) and the bulk temperature measured at depth.

### Common Structure (COARE and ECMWF variants)

1. First guess: Q_abs = Q_nsol (net non-solar surface heat flux)
2. Compute skin layer thickness δ from δ = f(α, Q_d, u*)
3. Iterate (4 times):
   a. Compute solar absorption fraction f_r through the skin layer
   b. Q_abs = Q_nsol + f_r · Q_sw
   c. Update δ
4. ΔT_cs = Q_abs · δ / k_water

### Key Differences

| Parameter | COARE | ECMWF |
|-----------|-------|-------|
| Solar absorption coefficient | 0.137 | 0.065 |
| Reference | Fairall et al. 1996 | Zeng & Beljaars 2005 |

---

## 7. Warm-Layer Parameterization

The warm-layer correction accounts for diurnal warming of the upper ocean layer
during daytime.

### COARE Warm-Layer (Fairall et al. 2019)

- **Prognostic depth**: H_wl = c_d1 · τ_ac / √(Q_ac), max 20 m
- **Temperature increment**: ΔT = c_d2 · Q_ac^1.5 / τ_ac
- **Dawn reset**: Resets accumulated heat/momentum at local dawn
- **Local solar time**: Computed from longitude and UTC time
- **3-band solar absorption model**: Exponential decay at different depth scales

### ECMWF Warm-Layer (Zeng & Beljaars 2005)

- **Fixed depth**: rd₀ = 3 m
- **Semi-implicit**: 10 iterations to converge
- **Stability function**: Takaya et al. (2010) — PHI(ζ)
  - Stable: rational function
  - Unstable: 1/√(1 + 16|ζ|)
- **Langmuir turbulence**: Optional, via Stokes drift parameterization
- **No dawn reset**: Uses continuous heat balance

---

## 8. Physical Constants

| Constant | Symbol | Value | Unit |
|----------|--------|-------|------|
| Gravity | g | 9.8 | m/s² |
| Von Kármán | κ | 0.4 | — |
| Gas constant (dry air) | R_dry | 287.05 | J/K/kg |
| Gas constant (water vapor) | R_vap | 461.495 | J/K/kg |
| Ratio R_dry/R_vap | ε | ~0.622 | — |
| Specific heat (dry air) | C_p,dry | 1005.0 | J/K/kg |
| Specific heat (water vapor) | C_p,vap | 1860.0 | J/K/kg |
| Specific heat (seawater) | C_p,w | 4190.0 | J/K/kg |
| Density of seawater | ρ_w | 1025.0 | kg/m³ |
| Emissivity (water) | ε_w | 0.98 | — |
| Emissivity (ice) | ε_i | 0.996 | — |
| Stefan-Boltzmann | σ | 5.67×10⁻⁸ | W/m²/K⁴ |
| Freezing point (fresh) | T_0 | 273.15 | K |
| Triple point | T_t | 273.16 | K |
| Kinematic viscosity (air) | ν_a | 1.5×10⁻⁵ | m²/s |
| Kinematic viscosity (water) | ν_w | 1.0×10⁻⁶ | m²/s |
| Thermal conductivity (water) | k_w | 0.6 | W/m/K |
| Reference pressure | P_atm | 101000.0 | Pa |
| Salt reduction factor | — | 0.98 | — |
| Max roughness (sea) | z₀_max | 0.0025 | m |
| Min transfer coefficient | C_x,min | 0.1×10⁻³ | — |

---

## 9. Humidity Conversions

The library accepts humidity in three forms:

### Specific Humidity (default, detected when values < 0.1)
Input is used directly as q [kg/kg]

### Relative Humidity (detected when values > 1)
q = q_air_rh(RH, T, P) = ε · e_sat(T) · RH / (P − (1−ε) · e_sat(T))

### Dew-Point Temperature (detected when values in plausible T range and not RH)
q = q_air_dp(T_d, P) = ε · e_sat(T_d) / (P − (1−ε) · e_sat(T_d))

### Saturation Vapor Pressure (Goff 1957)

Over water:
e_sat(T) = 10^(10.79574·(1−T_t/T) − 5.02800·log₁₀(T/T_t) − 1.50475×10⁻⁴·(1−10^(−8.2969·(T/T_t−1))) + 0.42873×10⁻³·(10^(4.76955·(1−T_t/T))−1) + 0.78614)

Over ice: Similar Goff formula with different coefficients.

---

## 10. Temperature Conversions

### Potential Temperature from Absolute Temperature
θ = T · (P₀/P)^(R_dry/C_p)

### Absolute Temperature from Potential Temperature
T = θ · (P/P₀)^(R_dry/C_p)

### Pressure at Height z (Barometric Equation, iterated 3 times)
P(z) from hydrostatic balance using virtual temperature

### Virtual Temperature
T_v = T · (1 + 0.608 · q)

### Moist Adiabatic Lapse Rate
Complex formula involving mixing ratio, see mod_phymbl for details.

---

## 11. Sea-Ice Extensions

The Andreas algorithm and additional ice-specific modules handle fluxes over
sea ice. Key differences from ocean:

- Fixed drag coefficient over ice: C_D,ice = 1.4×10⁻³ (default in NEMO)
- Latent heat uses L_sub (sublimation) instead of L_vap
- Emissivity: 0.996 (ice) vs 0.98 (water)
- Different roughness parameterizations (LKB, Lupkes, etc.)
- Ice fraction weighting for combined ocean+ice grids

### Ice-Specific Algorithms (from ice/ modules)

| Algorithm | Key Feature |
|-----------|-------------|
| NEMO default | Fixed C_D,ice, simple stability |
| Andreas 2005 | LKB roughness, u* polynomial |
| Lupkes et al. 2012 | Ice-concentration-dependent C_D |
| Lupkes & Gryanik 2015 | Parameterized form based on sea-ice morphology |
