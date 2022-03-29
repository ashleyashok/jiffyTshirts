-- How many customers have ever purchased a medium sized sweater with a discount ?
select
    count(DISTINCT(c.customer_uid))
from
    orders a
    join line_items b on (a.order_id = b.order_id)
    join customers c on (a.customer_uid = c.customer_uid)
where
    a.discount > 0
    and b.product_category = 'Sweater'
    and b.size = 'M'