package com.example.resource;

import com.example.resource.entity.Member;
import com.example.resource.repository.MemberRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.boot.CommandLineRunner;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

import java.time.LocalDate;

@Component
@RequiredArgsConstructor
public class DummyMemberLoader implements CommandLineRunner {

    private final MemberRepository memberRepository;

    @Override
    public void run(String... args) throws Exception {
        PasswordEncoder passwordEncoder = new BCryptPasswordEncoder();

        if (memberRepository.findByUsername("test").isEmpty()) {
            Member member = Member.builder()
                    .username("test")
                    .password(passwordEncoder.encode("1234"))
                    .name("홍길동")
                    .birthday(LocalDate.of(1990, 1, 1))
                    .userType("business")
                    .build();
            memberRepository.save(member);
        }
    }

}