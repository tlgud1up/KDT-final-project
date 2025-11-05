package com.example.resource.security;

import com.example.resource.entity.Member;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

import java.time.LocalDate;
import java.util.Arrays;
import java.util.Collection;

public class MemberDetails implements UserDetails {

    private final Member member;

    public MemberDetails(Member member) {
        this.member = member;
    }

    public Long getId() {
        return member.getId();
    }

    public String getName() {
        return member.getName();
    }

    @Override
    public String getUsername() {
        return member.getUsername();
    }

    @Override
    public String getPassword() {
        return member.getPassword();
    }

    public Member getMember() {
        return member;
    }

    public LocalDate getBirthday(){
        return member.getBirthday();
    }

    public LocalDate getJoinDate(){
        return member.getJoinDate();
    }

    public String getUserType(){
        return member.getUserType();
    }

    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
        return Arrays.stream(member.getRole().split(","))
                .map(SimpleGrantedAuthority::new)
                .toList();
    }

    @Override
    public boolean isAccountNonExpired() {
        return true;
    }

    @Override
    public boolean isAccountNonLocked() {
        return true;
    }

    @Override
    public boolean isCredentialsNonExpired() {
        return true;
    }

    @Override
    public boolean isEnabled() {
        return true;
    }
}
