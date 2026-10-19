with customers as (
    select * from {{ ref('stg_customers') }}
),

-- 先算几个后面要反复用到的基础量
base as (
    select
        *,
        max(enrolled_date) over ()                                   as reference_date,
        spend_wine + spend_fruit + spend_meat
            + spend_fish + spend_sweets + spend_gold                 as total_spent,
        purchases_web + purchases_catalog + purchases_store          as total_purchases,
        kids_at_home + teens_at_home                                 as children,
        case when marital_status in ('Married', 'Together')
             then 'Partner' else 'Alone' end                         as living_with,
        case when education in ('Master', 'PhD')       then 'Postgraduate'
             when education = 'Graduation'             then 'Graduate'
             else 'Undergraduate' end                                as education_group
    from customers
),

features as (
    select
        customer_id,

        -- 人口与家庭
        year(reference_date) - birth_year                            as age,
        education, education_group, marital_status, living_with,
        children,
        (children > 0)::int                                          as is_parent,
        children + case when living_with = 'Partner' then 2 else 1 end as family_size,

        -- 收入：保留原值，另给一个按学历组中位数填补的版本
        income,
        coalesce(income, median(income) over (partition by education_group)) as income_filled,

        -- 客户关系
        enrolled_date,
        date_diff('day', enrolled_date, reference_date)              as tenure_days,
        recency_days,

        -- 消费（过去 2 年累计）
        spend_wine, spend_fruit, spend_meat, spend_fish, spend_sweets, spend_gold,
        total_spent,
        round(spend_wine / nullif(total_spent, 0), 3)                as wine_share,
        round(spend_meat / nullif(total_spent, 0), 3)                as meat_share,
        round(spend_gold / nullif(total_spent, 0), 3)                as gold_share,

        -- 渠道
        purchases_web, purchases_catalog, purchases_store, purchases_deals, web_visits_month,
        total_purchases,
        round(purchases_web     / nullif(total_purchases, 0), 3)     as web_share,
        round(purchases_catalog / nullif(total_purchases, 0), 3)     as catalog_share,
        round(purchases_store   / nullif(total_purchases, 0), 3)     as store_share,
        round(purchases_deals   / nullif(total_purchases, 0), 3)     as deal_share,
        case
            when total_purchases = 0 then 'None'
            when purchases_store   >= greatest(purchases_web, purchases_catalog) then 'Store'
            when purchases_web     >= purchases_catalog                          then 'Web'
            else 'Catalog'
        end                                                          as preferred_channel,
        round(total_spent / nullif(total_purchases, 0), 1)           as avg_order_value,

        -- 营销活动（不含 responded，它是预测目标）
        accepted_cmp1 + accepted_cmp2 + accepted_cmp3
            + accepted_cmp4 + accepted_cmp5                          as total_campaigns_accepted,
        complained,
        responded,

        -- 数据质量标记（从 staging 带过来）
        is_income_missing, is_income_outlier, is_birth_year_invalid,
        is_inconsistent, is_possible_duplicate
    from base
),

-- RFM：每个维度按五分位打 1–5 分，5 分最好
rfm as (
    select
        *,
        ntile(5) over (order by recency_days desc, customer_id)      as r_score,
        ntile(5) over (order by total_purchases,   customer_id)      as f_score,
        ntile(5) over (order by total_spent,       customer_id)      as m_score
    from features
)

select
    *,
    r_score * 100 + f_score * 10 + m_score                           as rfm_code,
    -- 建模时默认使用的行：收入和出生年有效、没有逻辑矛盾
    (income is not null and age is not null and not is_inconsistent) as is_model_ready
from rfm