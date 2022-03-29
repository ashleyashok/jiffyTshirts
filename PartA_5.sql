select
    is_business,
    ((selling_rev - selling_cost) * 100 / selling_rev) as rate_of_return
from
    (
        SELECT
            is_business,
            sum(((selling_price *(1 - discount) * quantity))) as selling_rev,
            sum(((supplier_cost * quantity))) as selling_cost
        from
            orders a
            join line_items b on (a.order_id = b.order_id)
            join customers c on (a.customer_uid = c.customer_uid)
        group by
            1
    ) a