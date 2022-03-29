-- How profitable was our most profitable month? *Not including shipping costs and using UCT time for month
select
	year_month,
	selling_rev,
	selling_cost,
	(selling_rev - selling_cost) as profit
from
    (
        SELECT
            to_date(order_timestamp, 'YYYY-MM-00') as year_month,
            sum(((selling_price*(1-discount) * quantity))) as selling_rev,
            sum(((supplier_cost * quantity))) as selling_cost
        from
            orders a
            join line_items b on (a.order_id = b.order_id)
            join customers c on (a.customer_uid = c.customer_uid)
        group by
            1
    ) a
    
   order by profit desc 