
CREATE OR REPLACE TABLE keepcoding.ivr_summary AS

WITH previous_next_24H
  AS (SELECT ivr_id
      , phone_number
      , start_date
      , LAG(start_date) OVER (PARTITION BY phone_number ORDER BY start_date) AS previous_24H
      , LEAD(start_date) OVER (PARTITION BY phone_number ORDER BY start_date) AS next_24H
      , IF(DATETIME_DIFF(start_date, LAG(start_date) OVER (PARTITION BY phone_number ORDER BY start_date), HOUR) <= 24, 1, 0) AS repeated_phone_24H
      , IF(DATETIME_DIFF(LEAD(start_date) OVER (PARTITION BY phone_number ORDER BY start_date), start_date, HOUR)<= 24, 1, 0) AS cause_recall_phone_24H
    FROM `keepcoding.ivr_calls`)

SELECT detail.ivr_id
    , detail.phone_number
    , detail.ivr_result
    , CASE WHEN STARTS_WITH(detail.vdn_label, 'ATC') THEN 'FRONT'
          WHEN STARTS_WITH(detail.vdn_label, 'TECH') THEN 'TECH'
          WHEN detail.vdn_label = 'ABSORPTION' THEN 'ABSORPTION'
          ELSE 'RESTO'
     END AS vdn_aggregation
    , detail.start_date
    , detail.end_date
    , detail.total_duration
    , detail.customer_segment
    , detail.ivr_language
    , detail.steps_module
    , detail.module_aggregation
    , detail.document_type
    , detail.document_identification
    , detail.customer_phone
    , detail.billing_account_id
    , detail.module_name
    , IF(module_name = 'AVERIA_MASIVA', 1, 0) AS masiva_lg
    , detail.step_name
    , detail.step_description_error
    , IF(step_name = 'CUSTOMERINFOBYPHONE.TX' AND step_description_error = 'UNKNOWN', 1, 0) AS info_by_phone_lg
    , IF(step_name = 'CUSTOMERINFOBYDNI.TX' AND step_description_error = 'UNKNOWN', 1, 0) AS info_by_dni_lg
    , repeated_phone_24H
    , cause_recall_phone_24H

FROM `keepcoding.ivr_detail` detail
LEFT 
JOIN previous_next_24H
  ON detail.ivr_id = previous_next_24H.ivr_id
WHERE ((module_name = 'IDENTIFICACION' OR module_name = 'WELCOME') AND customer_phone = 'UNKNOWN'AND billing_account_id <> 'UNKNOWN')
  OR (module_name = 'WELCOME' AND customer_phone <> 'UNKNOWN'AND billing_account_id <> 'UNKNOWN')
  OR (module_name = 'AVERIA_MASIVA' AND step_name = 'GETINFOMASIVAS.TX')
 ORDER BY detail.ivr_id ASC
