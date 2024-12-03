# Visual Comparison of Distributional RL Methods

## Value Distribution Representations

### DQN (Point Estimate)

```
Q(s,a)
   │
   └── Single Value
       ▼
    Expected
     Return
```

### C51 (Fixed Categorical)

```
                 Probability
                    ▲
   █                │
   █      █         │
   █  █   █   █     │
   █  █   █   █  █  │
   █  █   █   █  █  │
───┴──┴───┴───┴──┴──┴─► Return
   z₁ z₂  z₃  z₄ z₅
  Fixed Support Points
```

### QR-DQN (Quantile-based)

```
                 Cumulative
                 Probability
                     ▲
                    1├─────────────████
                     │         ████
                     │     ████
                     │  ███
                    0├───╯
                     └┼─────────────┼─► Return
                      θ₁   θ₂   θ₃
                 Learned Quantiles
```

### IQN (Implicit Quantiles)

```
                 Quantile
                 Function
                     ▲
                    1├────────────╮
                     │         ╭──╯
                     │      ╭──╯
                     │   ╭──╯
                    0├───╯
                     └┼─────────────┼─► Return
                  Continuous Mapping
```

## Network Architectures

### DQN

```
Input         Hidden         Output
State ──► [Neural Net] ──► Q-values
 (s)                        Q(s,a)
```

### C51

```
Input         Hidden         Output
State ──► [Neural Net] ──► Probabilities
 (s)                     {p₁,...,pₙ} for
                        fixed {z₁,...,zₙ}
```

### QR-DQN

```
Input         Hidden         Output
State ──► [Neural Net] ──► Quantiles
 (s)                     {θ₁,...,θₙ} for
                        fixed {τ₁,...,τₙ}
```

### IQN

```
Input    Embedding       Output
State ──┬─► [Net] ───┬─► Quantile
 (s)    │            │   Values
        │            │
τ ~ U[0,1]           │
    └──► [Cos] ──────┘
```

## Evolution of Methods

```
     DQN           C51           QR-DQN         IQN
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │ Point        │    │ Fixed        │    │ Quantile     │
  │ Estimate     │ ─► │ Support      │ ─► │ Based        │
  │              │    │ Points       │    │ Values       │
  └──────────────┘    └──────────────┘    └──────────────┘
     │            │             │             │
     │            │             │             │
  Limited      Better        Flexible      Continuous
 Risk Info   Risk Info    Distribution    Mapping &
                                         Sampling
```

## Risk Sensitivity Comparison

```
Low Risk ◄────────────────────────────────────► High Risk

DQN:    [Fixed Risk - No Control]
C51:    [────── Discrete Control ──────]
QR-DQN: [──────── Fine Control ────────]
IQN:    [──── Continuous Control ──────]
```

## Computational Trade-offs

```
Memory Usage:        Low ◄─────────────────► High
DQN:     [█    ]
C51:     [████ ]
QR-DQN:  [███  ]
IQN:     [██   ]

Computation:        Fast ◄─────────────────► Slow
DQN:     [█    ]
C51:     [███  ]
QR-DQN:  [████ ]
IQN:     [███  ]

Sample Efficiency:   Low ◄─────────────────► High
DQN:     [██   ]
C51:     [███  ]
QR-DQN:  [████ ]
IQN:     [█████]
```

# Mathematical Details and Comparisons of Distributional RL Methods

## Value Distribution Representations

### DQN (Point Estimate)

\[ Q(s,a) = \mathbb{E}[R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \ldots] \]

### C51 (Categorical Distribution)

\[ Z_\theta(s,a) = \sum_{i=1}^N p_i(s,a) \delta_{z_i} \]

where:

- \( z_i = V_{min} + \frac{i-1}{N-1}(V_{max} - V_{min}) \)
- \( \sum_{i=1}^N p_i(s,a) = 1 \)

### QR-DQN (Quantile Distribution)

\[ Z_\theta(s,a) = \sum_{i=1}^N \frac{1}{N} \delta_{\theta_i(s,a)} \]

where:

- \( \tau_i = \frac{i - 0.5}{N} \)
- \( \theta_i(s,a) \) represents the \( \tau_i \)-quantile

### IQN (Implicit Quantile Function)

\[ Z_\theta(s,a,\tau) = f_\theta(s,a,\tau), \tau \sim U[0,1] \]

## Loss Functions

### DQN

\[ L_{DQN}(\theta) = \mathbb{E}[(r + \gamma \max_{a'} Q_{\theta^-}(s',a') - Q_\theta(s,a))^2] \]

### C51

\[ L_{C51}(\theta) = D_{KL}(\Phi T Z_{\theta'} \parallel Z_\theta) \]

where:

\[ T Z_{\theta'}(s,a) = r + \gamma Z_{\theta'}(s',a'^*) \]
\[ a^* = \arg\max_{a'} \mathbb{E}[Z_{\theta'}(s',a')] \]

### QR-DQN

\[ L_{QR}(\theta) = \mathbb{E}\left[\sum_{i=1}^N \sum_{j=1}^N \rho_{\tau_i}(r + \gamma \theta_j(s',a'^*) - \theta_i(s,a))\right] \]

where:

\[ \rho_\tau(u) = |\tau - \mathbb{1}\{u < 0\}| \cdot L_\kappa(u) \]

\[
L_\kappa(u) = \begin{cases}
\frac{1}{2}u^2 & \text{if } |u| \leq \kappa \\
\kappa(|u| - \frac{1}{2}\kappa) & \text{otherwise}
\end{cases}
\]

### IQN

\[ L_{IQN}(\theta) = \mathbb{E}*{\tau,\tau' \sim U[0,1]}\left[\sum*{i,j} \rho_{\tau_i}(r + \gamma Z_\theta(s',a'^*,\tau'*j) - Z*\theta(s,a,\tau_i))\right] \]

## Feature Embeddings

### IQN Cosine Embedding

\[ \phi_j(\tau) = \cos(\pi j \tau), j = 1,\ldots,n \]
\[ \psi(s,\tau) = \phi(s) \odot \phi(\tau) \]

## Comparison of Properties

| Property | DQN | C51 | QR-DQN | IQN |
|----------|-----|-----|---------|-----|
| Distribution Type | Point | Discrete | Discrete | Continuous |
| Parameters | O(\|A\|) | O(N\|A\|) | O(N\|A\|) | O(\|A\|) |
| Sample Complexity | High | Medium | Medium | Low |
| Memory Usage | Low | High | Medium | Low |
| Computational Cost | Low | Medium | High | Medium |

## Risk-Sensitive Policies

### C51

\[ \pi(s) = \arg\max_a \sum_{i=1}^N p_i(s,a)z_i \]

### QR-DQN

\[ \pi_\alpha(s) = \arg\max_a \frac{1}{N} \sum_{i=1}^N \theta_i(s,a) \]

### IQN

\[ \pi_\alpha(s) = \arg\max_a \mathbb{E}*{\tau \sim f*\alpha}[Z_\theta(s,a,\tau)] \]

where \( f_\alpha \) is a risk-sensitive sampling distribution.

## Distribution Updates

### C51 Projection

For target \( \hat{z} = r + \gamma z \):

\[ (\Phi \hat{z})*i = \begin{cases}
1 & \text{if } \hat{z} \leq z_1 \\
\frac{z*{i+1} - \hat{z}}{z_{i+1} - z_i} & \text{if } z_i < \hat{z} \leq z_{i+1} \\
0 & \text{if } \hat{z} > z_N
\end{cases} \]

### QR-DQN Update

\[ \theta_i(s,a) \leftarrow \theta_i(s,a) - \alpha \nabla_{\theta_i} \sum_j \rho_{\tau_i}(r + \gamma \theta_j(s',a'^*) - \theta_i(s,a)) \]

### IQN Update

\[ Z_\theta(s,a,\tau) \leftarrow Z_\theta(s,a,\tau) - \alpha \nabla_\theta \mathbb{E}*{\tau'}[\rho*\tau(r + \gamma Z_\theta(s',a'^*,\tau') - Z_\theta(s,a,\tau))] \]

## Evolution of Methods

```
DQN → C51 → QR-DQN → IQN
Point Est. → Fixed Dist. → Quantile Dist. → Continuous Dist.
```

## Computational Complexity

| Operation | DQN | C51 | QR-DQN | IQN |
|-----------|-----|-----|---------|-----|
| Forward Pass | O(\|A\|) | O(N\|A\|) | O(N\|A\|) | O(K\|A\|) |
| Loss Computation | O(1) | O(N) | O(N²) | O(K²) |
| Memory | O(\|A\|) | O(N\|A\|) | O(N\|A\|) | O(K\|A\|) |

where:

- $\|A\|$ is the action space size
- $N$ is the number of atoms/quantiles
- $K$ is the number of sampled quantiles in IQN (typically $K < N$)
