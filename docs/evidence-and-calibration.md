# Evidence and calibration note

The repository does not contain a sufficiently documented, point-in-time historical cohort with survivorship controls, source availability snapshots, transaction costs, and out-of-sample validation. Historical predictive calibration is therefore unavailable.

The default scanner treats its numeric screening score only as an explainable ordering mechanism after hard safety/evidence gates. It must not be interpreted as the probability of a price multiple, a positive expected-value claim, or a measured trading edge.

The following contextual signals are retained for research but are neutral in the default classifier:

- narrative/name keywords;
- deployer or creator identity by itself;
- celebrity names, links, or generic social buzz;
- DEXScreener paid boosts.

A celebrity association is identity evidence only when an exact canonical account’s evidence contains the exact mint. Even verified identity evidence does not imply price potential and cannot bypass age, liquidity, trading-flow, on-chain authority/extension, holder, rug, or scam-evidence checks.

Future calibration would require immutable observation-time features for every discovered candidate (including rejected and deferred candidates), explicit source/evidence availability, predefined outcomes and horizons, leakage-resistant train/test splits, and reporting of uncertainty and base rates. The `candidate_observations` table is intended to support creation of such future cohorts; it does not itself establish predictive validity.
