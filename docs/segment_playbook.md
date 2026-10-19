# Customer Segment Playbook

> Knowledge base for the marketing assistant (RAG source).
> Data: Kaggle "Customer Personality Analysis" public dataset, 2,205 model-ready customers, enrolled Jul 2012 – Jun 2014.
> Spend figures are cumulative over the last 2 years. All numbers come from `customer_features` and `customer_segments`.
> Strategies in this document are recommendations derived from the data. They have not been tested in a live campaign and should be validated with an A/B test before roll-out.

---

## 1. Overview of the four segments

Customers were grouped with K-Means (K = 4) on eight features: income, age, total spend, number of children, tenure, total purchases, monthly web visits and recency. K = 4 was chosen over K = 2 (higher silhouette score) because four groups are easier to act on in marketing.

| Segment | Customers | Share of customers | Share of revenue | Avg income | Avg age | Response rate |
|---|---|---|---|---|---|---|
| Affluent High-Value | 539 | 24.4% | 52.4% | $76,767 | 45 | 24.1% |
| Established Families | 495 | 22.4% | 35.0% | $60,896 | 51 | 17.4% |
| Price-Sensitive New | 581 | 26.3% | 6.8% | $42,863 | 49 | 4.3% |
| Young Online Browsers | 590 | 26.8% | 5.8% | $30,806 | 37 | 15.6% |

Key takeaway: the two higher-income segments (47% of customers) generate 87% of revenue. The overall response rate to the last campaign is 15.1%.

---

## 2. Segment: Affluent High-Value

**Who they are.** Highest income (average $76,767), mostly without children at home (0.21 children on average; only 20% are parents). Average age 45. 38% hold a postgraduate degree.

**How they buy.**
- Highest spend: $1,307 on average over 2 years, with an average order value of $70 — about four times the two lower segments.
- About 19 purchases in 2 years, split across store (44%), catalog (30%) and web (26%). This is the only segment where catalog is a major channel.
- Rarely buy on discount: only 6.6% of purchases use a deal.
- Lowest web visits (2.4 per month) — they come to buy, not to browse.
- Spend mix: wine 44%, meat 34%, fish/fruit/sweets 17%, gold products 6%.

**Campaign history.** Most responsive segment. Response rate to the last campaign is 24.1%. 41% accepted at least one earlier campaign; campaign 5 (25%) and campaign 1 (20%) worked especially well with them.

**Recommended strategy: retain and grow share of wallet.**
- Treat as VIPs: premium bundles (wine + meat pairings), early access to new products, loyalty tiers.
- Use catalog and personalised email; do not rely on discount offers, which they rarely need and which erode margin.
- Re-use the creative and offer style of campaigns 1 and 5.
- Watch recency: customers who bought within 30 days respond at 34%, versus 16% for those inactive for 60+ days. Trigger a win-back message when recency passes 60 days.

**KPIs.** Retention rate, average order value, response rate, revenue per customer.

---

## 3. Segment: Established Families

**Who they are.** Oldest segment (average age 51) with solid income ($60,896). Most are parents (84%), typically with teenagers. Longest-standing customers: average tenure 504 days.

**How they buy.**
- Second-highest spend: $949 on average, average order value $51, about 19 purchases in 2 years.
- Channels: store 44%, web 35%, catalog 21%.
- Most deal purchases of any segment (3.5 per customer; 19% of purchases use a deal).
- Wine is 59% of their spend — the highest share of any segment.
- Moderate web activity (5.9 visits per month).

**Campaign history.** Response rate 17.4%. 24% accepted an earlier campaign; campaign 4 worked best (14%).

**Recommended strategy: loyalty and value bundles.**
- Reward tenure: loyalty points, anniversary offers, "thank you for N years" messages.
- Family-size and multi-buy bundles, especially wine; deals work here but should be targeted rather than blanket.
- Balance store and web — they already use both.
- Recency matters less in this segment (22% response if active within 30 days vs 14% if inactive 60+ days), so regular, steady contact is more useful than urgent win-back.

**KPIs.** Purchase frequency, deal redemption rate, retention, revenue per customer.

---

## 4. Segment: Price-Sensitive New

**Who they are.** Middle-to-lower income ($42,863), the most children at home (1.5 on average; 95% are parents). Average age 49. Newest customers: average tenure only 195 days.

**How they buy.**
- Low spend: $157 on average (median $84), average order value $17, about 7 purchases in 2 years.
- Mostly in store (52%), then web (36%); catalog is rarely used (11%).
- Highly deal-driven: 33% of purchases use a discount.
- Spend mix: wine 58%, meat 19%, gold products 11%.

**Campaign history.** Least responsive segment: 4.3% response to the last campaign. Only 11% accepted any earlier campaign and none accepted campaign 5. Even recent buyers respond at only 10%.

**Recommended strategy: low-cost nurture; do not over-invest.**
- Expensive campaigns are unlikely to pay back. Prefer low-cost channels such as email, in-store promotions and receipts.
- Lead with value: family packs, everyday-low-price messaging, coupons tied to the next store visit.
- Focus on building the habit of a second and third purchase while they are new, rather than on one-off campaign responses.
- Use the response model to pick the few customers in this segment with high predicted probability instead of targeting the whole segment.

**KPIs.** Second-purchase rate, coupon redemption, cost per acquisition, campaign ROI.

---

## 5. Segment: Young Online Browsers

**Who they are.** Youngest segment (average age 37) with the lowest income ($30,806). Most are parents (85%), usually with young children. Average tenure 442 days.

**How they buy.**
- Lowest spend: $132 on average (median $71), average order value $17, about 6.5 purchases in 2 years.
- Heaviest web users: 7.3 web visits per month (the highest) and 40% of purchases online — but few visits turn into purchases.
- Most deal-driven segment: 35% of purchases use a discount.
- Highest share of gold products (17%) and of fish/fruit/sweets (19%) in their spend.

**Campaign history.** Despite low spend, they respond at 15.6%, close to the overall average. Campaign 3 was the only earlier campaign that worked for them (9%). Recency matters a lot: 29% response if they bought within 30 days, but only 7% if inactive for 60+ days.

**Recommended strategy: improve online conversion and strike while active.**
- The opportunity is the gap between many web visits and few web purchases: simplify checkout, use abandoned-cart reminders, show personalised product recommendations.
- Time offers to recent activity — target customers within 30 days of their last purchase.
- Use digital channels (email, app, social) and time-limited deals; reuse the approach of campaign 3.
- Do not expect large orders; measure success by conversion rate and frequency.

**KPIs.** Web conversion rate (purchases per visit), response rate among recent buyers, purchase frequency.

---

## 6. How to use the response model

The `/predict` endpoint returns `response_probability`: the model's estimated probability that a customer accepts the next campaign.

- **Model:** XGBoost, chosen over logistic regression and random forest by 5-fold cross-validation.
- **Performance on held-out test data:** ROC-AUC 0.90, PR-AUC 0.68 (a random guess would score about 0.15).
- **Business meaning:** contacting the top 20% of customers by predicted probability reaches about 70% of the customers who actually respond.
- **`will_respond`** is simply `response_probability >= 0.5`. For campaign targeting, ranking customers by probability and choosing a budget-based cut-off (for example the top 20%) is usually better than the fixed 0.5 threshold.

Suggested decision rules:
- Probability ≥ 0.5 → include in the campaign; for Affluent High-Value, use the premium offer.
- Probability 0.2–0.5 → include if budget allows; prefer low-cost channels.
- Probability < 0.2 → exclude from paid campaigns; keep in low-cost newsletters only.

---

## 7. Cross-segment insights

- **Past behaviour predicts future response.** Customers who accepted any of the five earlier campaigns responded at 40.8%, versus 8.4% for those who never accepted one — almost five times higher.
- **Recency is a strong signal.** Response rises steadily with recency score: from about 6.5% for the least recent fifth of customers to about 29% for the most recent fifth.
- **Discounts are not the main driver of value.** The segment that uses deals least (Affluent High-Value, 6.6%) spends the most and responds best.
- **Store is the main channel for every segment** (44–52% of purchases). Web matters most for Young Online Browsers; catalog matters only for Affluent High-Value.

---

## 8. Data limitations

- This is a public demonstration dataset, not a real company's customer base; the data covers 2012–2014.
- It is unclear whether `Recency` was measured before or after the last campaign. If after, its predictive power may be overstated.
- The dataset does not define the education level "2n Cycle"; it is grouped with Basic as Undergraduate because its income and spend sit below Graduation.
- 35 customers were excluded from modelling (missing or invalid income or birth year, or inconsistent purchase records). 358 rows are flagged as possible duplicates and were kept.
- Strategies in this document are hypotheses and should be validated with controlled experiments. As a reference, detecting a lift in response rate from 15.6% to 20% with 80% power at a 5% significance level needs roughly 600 customers per group (two-sided test).

---

## 9. Glossary

- **Response rate:** share of customers who accepted the last campaign (`responded = 1`).
- **Recency:** days since the customer's last purchase; lower is more recent.
- **Tenure:** days since the customer enrolled, measured to 29 June 2014.
- **Average order value (AOV):** total spend divided by total purchases.
- **Deal share:** share of purchases made with a discount.
- **Share of revenue:** a segment's share of total 2-year spend.
