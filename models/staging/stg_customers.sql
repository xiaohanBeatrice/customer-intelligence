with raw as (
    select * from read_csv('data/raw/marketing_campaign.csv',
                           delim='\t', header=true, dateformat='%d-%m-%Y')
),

-- 第一层：只改列名
renamed as (
    select
        ID                  as customer_id,
        Year_Birth          as birth_year_raw,
        Education           as education_raw,
        Marital_Status      as marital_status_raw,
        Income              as income_raw,
        Kidhome             as kids_at_home,
        Teenhome            as teens_at_home,
        Dt_Customer         as enrolled_date,
        Recency             as recency_days,
        MntWines            as spend_wine,      -- 金额，过去 2 年累计
        MntFruits           as spend_fruit,
        MntMeatProducts     as spend_meat,
        MntFishProducts     as spend_fish,
        MntSweetProducts    as spend_sweets,
        MntGoldProds        as spend_gold,
        NumDealsPurchases   as purchases_deals,
        NumWebPurchases     as purchases_web,
        NumCatalogPurchases as purchases_catalog,
        NumStorePurchases   as purchases_store,
        NumWebVisitsMonth   as web_visits_month,
        AcceptedCmp1        as accepted_cmp1,
        AcceptedCmp2        as accepted_cmp2,
        AcceptedCmp3        as accepted_cmp3,
        AcceptedCmp4        as accepted_cmp4,
        AcceptedCmp5        as accepted_cmp5,
        Complain            as complained,
        Response            as responded
        -- Z_CostContact、Z_Revenue 是常量列，不选
    from raw
)

-- 第二层：清洗 + 打标记
select
    customer_id,

    -- 出生年：早于 1920 视为录入错误
    case when birth_year_raw < 1920 then null else birth_year_raw end as birth_year,
    birth_year_raw < 1920                                             as is_birth_year_invalid,

    -- 学历：保留原值，合并放到 marts 层（2n Cycle 的含义数据集未定义）
    education_raw                                                     as education,

    -- 婚姻状态：Alone / Absurd / YOLO 归为 Single
    case
        when marital_status_raw in ('Alone', 'Absurd', 'YOLO') then 'Single'
        else marital_status_raw
    end                                                               as marital_status,

    -- 收入：666666 改为空值；高于 IQR 上限的打标记但保留
    case when income_raw > 200000 then null else income_raw end       as income,
    income_raw is null                                                as is_income_missing,
    income_raw > 118351                                               as is_income_outlier,

    kids_at_home, teens_at_home, enrolled_date, recency_days,
    spend_wine, spend_fruit, spend_meat, spend_fish, spend_sweets, spend_gold,
    purchases_deals, purchases_web, purchases_catalog, purchases_store, web_visits_month,
    accepted_cmp1, accepted_cmp2, accepted_cmp3, accepted_cmp4, accepted_cmp5,
    complained, responded,

    -- 逻辑矛盾：没有购买却有消费，或者折扣购买次数多于总购买次数
    (purchases_web + purchases_catalog + purchases_store = 0
        and spend_wine + spend_fruit + spend_meat + spend_fish + spend_sweets + spend_gold > 0)
    or purchases_deals > purchases_web + purchases_catalog + purchases_store
                                                                      as is_inconsistent,

    -- 疑似重复：除 ID 外所有字段都相同
    count(*) over (partition by
        birth_year_raw, education_raw, marital_status_raw, income_raw, kids_at_home,
        teens_at_home, enrolled_date, recency_days, spend_wine, spend_fruit, spend_meat,
        spend_fish, spend_sweets, spend_gold, purchases_deals, purchases_web,
        purchases_catalog, purchases_store, web_visits_month, accepted_cmp1, accepted_cmp2,
        accepted_cmp3, accepted_cmp4, accepted_cmp5, complained, responded
    ) > 1                                                             as is_possible_duplicate

from renamed