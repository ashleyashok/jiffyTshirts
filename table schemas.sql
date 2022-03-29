ALTER TABLE "public"."customers"
ADD PRIMARY KEY ("customer_uid");

ALTER TABLE "public"."orders"
ADD PRIMARY KEY ("order_id");

ALTER TABLE "public"."orders" ADD FOREIGN KEY ("customer_uid") REFERENCES "public"."orders" ();

ALTER TABLE "public"."orders" ALTER COLUMN "order_timestamp" SET DATA TYPE TIMESTAMP WITH TIME ZONE USING order_timestamp::timestamp with time zone;
