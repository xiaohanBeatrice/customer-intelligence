-- 返回不合理的行；返回 0 行 = 测试通过
select customer_id, age, total_spent, tenure_days
from {{ ref('customer_features') }}
where age not between 18 and 100
   or total_spent < 0
   or tenure_days < 0