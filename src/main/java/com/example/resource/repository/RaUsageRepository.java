package com.example.resource.repository;

import com.example.resource.entity.RaUsage;
import org.springframework.data.jpa.repository.JpaRepository;

public interface RaUsageRepository extends JpaRepository<RaUsage, Long> {
}