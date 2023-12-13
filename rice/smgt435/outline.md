
# SMGT 435: Baseball Analytics

## Week 1: Pythagorean Formula
- Bill James originally introduced it
- We can demonstrate that it predicts winning percentage better than winning percentage does
  - And we can do even better with $\alpha \neq 0$.
- And the residual is mostly noise
- So the implication is that we should evaluate teams on the basis of runs and ignore actual record
- What reasons might a team acutally be better than their run differential?
- Ideas: Bias-variance tradeoff, regression to the mean, true talent
- BUT if you had infinite sample size, you would prefer to use actual record
- So at what point (sample size) would we prefer winning percentage over Pythag?

In R:
- Calculate best $\alpha$
- Calculate split-half correlation for winning percentage, Pythag, residual

## Week 2: Base-Out Run Expectancy, Linear Weights

- Last week we discussed where wins come from (runs). But where do runs come from?
- Introduce Markov chains
- Talk about why a Markov chain is a good model for a baseball inning
- Define base-out run expectancy
- Define RE24
- Define linear weights
- Clicker Question: Which method would you prefer for evaluating hitters?
- Clicker Question: Can you explain how these metrics relate to each other in terms of the bias-variance tradeoff?
- So if you had infinite sample, you would prefer RE24
- At what sample size would you switch to preferring LW?

## Week 3: Batted Ball Outcome Model

- Last week we discussed where runs come from (plate appearance outcomes). But where do plate appearance outcomes come from?
- Discuss technology for measure off-bat characteristics
- Introduce expected linear weight
  - xLW = E[LW | batted ball characteristics]
  - Talk about how to model this expectation
- What is the tradeoff between using LW and using xLW?
- So if you had infinite sample size, you would prefer LW
- What is the tradeoff between including vs. not including horizontal angle?

[[What reasons might an outlier LW - xLW be attributable to signal rather than noise?]]

## Week 4: Fielding, Baserunning

- So now that we've estimated the batted ball outcome model, a lot of the residual is going to be due to the fielders
- Let's exted the batted ball outcome model to evaluate the contributions of fielders
- We have this residual, and we want to distribute it across the fielders involved in the play
- We'll simplify it by reducing it to a question of whether the out was made
- The big thing we're missing is the impact of outfield arm
- Let's evaluate that and baserunners
- For baserunners, let's also evaluate base stealing (and do the same for pitchers and catchers)

## Week 5: BABIP, FIP and DIPS

- BABIP is a popular statistic. How does one use it?
  - How does this connect with what we've already talked about?
- So there's this recognition that the signal-to-noise ratio for batted ball outcomes is lower than for other outcomes
- For pitchers, this acknowledgment is reflected by FIP. Who can explain what FIP is?
- FIP is an example of DIPS theory, i.e. Defense-Independent Pitching Statistics. The theory, popularized by Voros McCracken in 199?, says that when evaluating pitchers, you're better off completely ignoring batted ball outcomes because (a) they're noisy and (b) they're more attributable to the defense
- After how many plate appearances do you prefer giving full weight to BIP outcomes over giving zero weight to BIP outcomes?
- The answer is always somewhere in the middle. Let's find it via regression to the mean (homework?)

## Week 6: Count Value, Catcher Framing

- We talked about base-out run expectancy and linear weights. Let's get a little more granular and discuss count value, which will allow us to evaluate changes in run expectancy pitch-by-pitch
- Let's model the progression of a plate appearance as a Markov chain and calculate the run value of each count
- Application 1: So now we can evaluate the effectiveness of different pitch types (MUCH better than batting average against)
- Application 2: Pitch framing
  - Need to build a regression model for strike probability
  - Now we can put a run value on a called strike

## Week 7: Pitch Outcome Model

- Introduce pitch tracking data
- We want to predict the probability of each outcome. Why?
  - What are the possible outcomes?
  - Introduce binary outcome tree
- Estimate the models

## Week 8: Replacement Level, Positional Adjustment, WAR

- We've evaluated batting, fielding, baserunning and pitching. Now let's put them all together!
- Example questions:
  - Would you rather have the 2023 performance of Yandy Diaz (164 wRC+ as first baseman, really bad baserunning), Cal Raleigh (111 wRC+ as catcher, average baserunning) or Blake Snell (180 innings, 2.25 ERA, 3.44 FIP)?
  - Would you rather have the 2023 performance of Mitch Keller (194.1 innings, 4.21 ERA, 3.80 FIP), Justin Verlander (162.1 innings, 3.22 ERA, 3.85 FIP) or Tarik Skubal (80.1 innings, 2.80 ERA, 2.00 FIP)?
  - Sandy Alcantara (184.2 innings, 4.14 ERA, 4.03 FIP as starter) or Tanner Scott (78.0 innings, 2.31 ERA, 2.17 FIP as reliever)?
- Now that we've measured all of these skills on the run scale, we can make apples-to-oranges comparisons
- We still need the concept of replacement level to compare players with disparate amounts of playing time
- And how do we account for the fact that some positions are harder than others?
- This part of WAR is where theory is weakest
- Discuss offense-based position adjustments vs. defense-based position adjustments
- Discuss history of position adjustments used on FanGraphs
- Discuss replacement level used by FanGraphs and Baseball Reference
- Discuss how a team would determine replacement level for internal use

## Week 9: Guest Speakers

## Week 10: Projections
- Introduce MARCEL
- How would MARCEL be different if we were projecting K% instead of wOBA? 2B%?
- There are two things going on here: How noisy is the metric? And how much do players change from year to year?
- Can we use linear regression to estimate the MARCEL coefficients?
- Discuss grid search

## Week 11: Aging Curves, Level Translations
- Two different methods for aging curves: regression and delta method
  - How does selection bias affect each method?
- How to interpret minor league performance?
- Additive level translations
  - How does selection bias affect these estimates?
- Homework: Create projections for minor league players

## Week 12: Contract Valuation, Stuff, In-Game Decision-making

## Week 13: Guest Speakers

## Week 14: Student Presentations

