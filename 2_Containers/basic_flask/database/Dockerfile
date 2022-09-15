FROM postgres:9.4
ENV POSTGRES_PASSWORD=mysecretpassword
ENV POSTGRES_USER=myusername
ENV POSTGRES_DB=database
COPY init_table.sql /docker-entrypoint-initdb.d/init_table.sql