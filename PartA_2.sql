-- 2. How many orders were completed in 2018 containing at least 10 units?
select
    count(DISTINCT(a.order_id))
from
    orders a
    join line_items b on (a.order_id = b.order_id)
where
    quantity >= 10