package com.example.resource.repository;

import com.example.resource.entity.Member;
import com.example.resource.entity.OrigImage;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface OrigImageRepository extends JpaRepository<OrigImage, Long> {
    List<OrigImage> findAllByMemberOrderByAnalysisDateDesc(Member member);
}
