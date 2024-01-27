This method is based on an approximation formula for log return. The following is the explanation of the formula:

Suppose we have generated $M$ samples of pools, the return of each scenario is $\{r_j\}_{j=1}^M$. Assume these are sorted from small to large. Then the CVaR is $\frac{1}{M_{\alpha}}\sum_{j=1}^{M_{\alpha}}r_j$, where $M_{\alpha}$ is the index corresponding to the desired quantile. Then the derivative of CVaR can be obtain as the sum of the derivative of the return.

Now let's derive the derivative of the returns.

Suppose initially the X coins and Y coins of each pool are $\{(R_{X,0}^i,R_{Y,0}^i)\}_{i=1}^{N}$, and the weights of investment are $\theta =(\theta_1,\dots,\theta_N)^\top$, where $\sum_{i=1}^N\theta_i = 1$. Because the X coins we owned at the beginning are small compared to the reserve in each pool, we can approximately assume the increment of X coin in each pool is $\{\frac{1}{2}x_0\theta_i\}_{i=1}^N$, where $x_0$ is the total amount of X coins we owned at the beginning. Thus the LP coins in each pool are about $\{\frac{\frac{1}{2}x_0\theta_i}{R^i_{X,0}} \}_{i=1}^N$, given the initial LP coins being 1. Let's denote $\{\frac{\frac{1}{2}x_0\theta_i}{R^i_{X,0} + \frac{1}{2}x_0\theta_i} \}_{i=1}^N$ by $\{Q_i\}_{i=1}^N$. 

Assume the Reserve of pools at the end time T are $\{(R_{X,T}^i,R_{Y,T}^i)\}_{i=1}^{N}$, then the X coins and Y coins are $\{(R_{X,T}^iQ_i,R_{Y,T}^iQ_i)\}_{i=1}^{N}$, denoted by $\{(X_T^{i,\theta_i},Y_T^{i,\theta_i})\}_{i=1}^{N}$. To swap Y to X, we need to compare the X coins we can gain from each pool, which is $\frac{(1-\phi_i)R_{X,T}^i}{R_{Y,T}^i + (1 - \phi_i)\sum_{j=1}^N( Y_T^{j,\theta_j}) - Y_T^{i,\theta_i}} \times\sum_{j=1}^N( Y_T^{j,\theta_j})$

Thus the total X coins we gain in the end are $\sum_{i=1}^N(X_T^{i,\theta_i}) + \max_{1\leq i \leq N}\{\frac{(1-\phi_i)R_{X,T}^i}{R_{Y,T}^i + (1 - \phi_i)\sum_{j=1}^N( Y_T^{j,\theta_j}) - Y_T^{i,\theta_i}}\} \sum_{j=1}^N( Y_T^{j,\theta_j})$

It's a rather complicated formula, only an approximate derivative can be obtained as it's meaningless to get a precise derivative from an estimation. So at this point, we assume the main contribution of the derivative comes from the derivative of $\{Q_i\}_{i=1}^N$. Thus, the derivative w.r.t $\theta_i$ is $R_{X,T}^i\frac{\partial}{\partial \theta_i} Q_i + \max_{1\leq i \leq N}\{\frac{(1-\phi_i)R_{X,T}^i}{R_{Y,T}^i + (1 - \phi_i)\sum_{j=1}^N( Y_T^{j,\theta_j}) - Y_T^{i,\theta_i}}\} R_{Y,T}^i\frac{\partial}{\partial \theta_i} Q_i$

Then using the chain rule, we can get the derivative of log return.