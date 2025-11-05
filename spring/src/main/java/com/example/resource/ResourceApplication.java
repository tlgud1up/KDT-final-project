package com.example.resource;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.io.InputStream;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Properties;

@Slf4j
@SpringBootApplication
public class ResourceApplication {

    public static void main(String[] args) {

        Properties props = new Properties();
        try (InputStream input = ResourceApplication.class.getClassLoader()
                .getResourceAsStream("application.properties")) {
            if (input == null) {
                log.error("application.properties 파일을 찾을 수 없음");
                return;
            }
            props.load(input);
        } catch (IOException e) {
            log.error("application.properties 로드 실패 : {}", e.getMessage());
            return;
        }

        String dataSourceUrl = props.getProperty("spring.datasource.url");
        String username = props.getProperty("spring.datasource.username");
        String password = props.getProperty("spring.datasource.password");
        String dbName = props.getProperty("spring.datasource.dbname");

        String baseUrl = dataSourceUrl.replace("/" + dbName, "");

        try (Connection conn = DriverManager.getConnection(baseUrl, username, password);
             Statement stmt = conn.createStatement()) {
            stmt.executeUpdate("CREATE DATABASE IF NOT EXISTS " + dbName +
                    " CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;");
        } catch (SQLException e) {
            log.error("DB 생성 실패: {}", e.getMessage());
        }

        SpringApplication.run(ResourceApplication.class, args);
    }
}
